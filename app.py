"""EarlyAlert - Academic Risk Early-Warning Agent.  Run: streamlit run app.py
pip install streamlit pandas numpy plotly requests"""
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, sqlite3, os, json, hashlib, datetime as dt
DB = "earlyalert.db"
H = lambda p: hashlib.sha256(p.encode()).hexdigest()
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, role TEXT, name TEXT);
CREATE TABLE IF NOT EXISTS students(id TEXT PRIMARY KEY, name TEXT, department TEXT, semester INTEGER, mentor_id INTEGER REFERENCES users(id));
CREATE TABLE IF NOT EXISTS subjects(id INTEGER PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS attendance(id INTEGER PRIMARY KEY, student_id TEXT REFERENCES students(id), subject_id INTEGER REFERENCES subjects(id), week INTEGER, pct REAL);
CREATE TABLE IF NOT EXISTS internal_marks(id INTEGER PRIMARY KEY, student_id TEXT REFERENCES students(id), subject_id INTEGER REFERENCES subjects(id), week INTEGER, marks REAL);
CREATE TABLE IF NOT EXISTS assignments(id INTEGER PRIMARY KEY, student_id TEXT REFERENCES students(id), subject_id INTEGER REFERENCES subjects(id), week INTEGER, submitted INTEGER, missed INTEGER);
CREATE TABLE IF NOT EXISTS weekly_performance(id INTEGER PRIMARY KEY, student_id TEXT REFERENCES students(id), week INTEGER, score REAL);
CREATE TABLE IF NOT EXISTS risk_analysis(id INTEGER PRIMARY KEY, student_id TEXT REFERENCES students(id), week INTEGER, score REAL, category TEXT, factors TEXT, explanation TEXT, source TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS mentor_notes(id INTEGER PRIMARY KEY, student_id TEXT REFERENCES students(id), text TEXT, proposed_slot TEXT, status TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS approval_history(id INTEGER PRIMARY KEY, note_id INTEGER REFERENCES mentor_notes(id), action TEXT, by_user TEXT, at TEXT);
"""
def run(sql, p=()):
    with sqlite3.connect(DB) as c:
        c.execute("PRAGMA foreign_keys=ON"); cur = c.execute(sql, p); return cur.lastrowid
def df(sql, p=()):
    with sqlite3.connect(DB) as c: return pd.read_sql_query(sql, c, params=p)
now = lambda: dt.datetime.now().strftime("%Y-%m-%d %H:%M")

# ---------------- risk engine (transparent: 40% attendance, 35% marks, 25% assignments + trend adjustments up to +20) -------------
def compute_risk(sid, wk):
    a = df("SELECT s.name sub,week,pct FROM attendance a JOIN subjects s ON s.id=subject_id WHERE student_id=? AND week<=?", (sid, wk))
    m = df("SELECT s.name sub,week,marks FROM internal_marks a JOIN subjects s ON s.id=subject_id WHERE student_id=? AND week<=?", (sid, wk))
    s = df("SELECT week,submitted,missed FROM assignments WHERE student_id=? AND week<=?", (sid, wk))
    aw, mw = a.groupby("week").pct.mean(), m.groupby("week").marks.mean()
    att, marks = aw.iloc[-1], mw.tail(2).mean()
    att0, marks0 = aw.iloc[max(len(aw) - 4, 0)], mw.iloc[max(len(mw) - 4, 0)]
    s4 = s[s.week > wk - 4]; sub, mis = int(s4.submitted.sum()), int(s4.missed.sum())
    rate = sub / max(sub + mis, 1)
    ar = float(np.clip((90 - att) / 40 * 100, 0, 100)); mr = float(np.clip((75 - marks) / 45 * 100, 0, 100)); sr = float(np.clip((1 - rate) * 150, 0, 100))
    base = .40 * ar + .35 * mr + .25 * sr; adj, flags = 0, []
    if att0 - att >= 5: adj += 5 + (3 if att0 - att >= 10 else 0); flags.append(f"Attendance fell from {att0:.0f}% to {att:.0f}% over recent weeks")
    ml = mw.tail(3).values
    if len(ml) == 3 and ml[0] > ml[1] > ml[2] and ml[0] - ml[2] >= 5: adj += 5; flags.append(f"Internal marks fell in consecutive assessments ({ml[0]:.0f} -> {ml[2]:.0f})")
    sw = s.groupby("week").missed.sum().tail(3)
    if (sw > 0).sum() >= 2: adj += 4; flags.append("Assignments missed in repeated weeks")
    la, lm = a[a.week == wk].set_index("sub").pct, m[m.week == wk].set_index("sub").marks
    poor = [x for x in la.index if la[x] < 75 or (x in lm.index and lm[x] < 45)]
    if len(poor) >= 2: adj += 3; flags.append("Weak performance in multiple subjects: " + ", ".join(poor))
    score = round(min(100, base + adj), 1); cat = "High" if score >= 70 else "Medium" if score >= 40 else "Low"
    return dict(score=score, cat=cat, att=att, att0=att0, marks=marks, marks0=marks0, missed=mis, rate=rate, flags=flags, poor=poor,
                parts=dict(attendance=round(.4 * ar, 1), marks=round(.35 * mr, 1), assignments=round(.25 * sr, 1), trend=adj))
def rule_text(name, r):
    t = [f"{name} is classified as {r['cat']} Risk (score {r['score']}/100)."]
    t.append(f"Attendance is {r['att']:.0f}% (was {r['att0']:.0f}% four weeks ago); internal marks average {r['marks']:.0f} (was {r['marks0']:.0f}); {r['missed']} assignment(s) missed in the last 4 weeks.")
    if r["flags"]: t.append("Main factors: " + "; ".join(r["flags"]) + ".")
    t.append("This is based only on recorded data; it does not explain personal causes." if r["cat"] != "Low" else "No significant concerns in the recorded data.")
    return " ".join(t)
def recs(r):
    out = []
    if r["cat"] == "High": out.append("Schedule an individual mentoring session")
    if r["att"] < 75: out.append("Agree an attendance improvement plan")
    if r["marks"] < 50 or r["poor"]: out.append("Recommend remedial classes" + (f" in {', '.join(r['poor'])}" if r["poor"] else ""))
    if r["missed"] > 0: out.append("Encourage regular assignment submission with weekly check-ins")
    out.append("Share additional study resources"); return out
def llm_text(name, r, key, model):
    import requests
    facts = {k: r[k] for k in ("score", "cat", "att", "att0", "marks", "marks0", "missed", "flags", "parts")}
    msg = f"Write a 3-sentence explanation for a mentor about why student {name} is at risk. Use ONLY these facts, no assumptions about personal reasons, then suggest support: {json.dumps(facts, default=float)}"
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                        json={"model": model, "temperature": .2, "messages": [{"role": "user", "content": msg}]}, timeout=15)
    res.raise_for_status(); return res.json()["choices"][0]["message"]["content"].strip()
def next_slot():
    d = dt.date.today() + dt.timedelta(days=2)
    while d.weekday() > 4: d += dt.timedelta(days=1)
    return f"{d:%a %d %b %Y}, 4:00 PM (suggested - not confirmed)"
def draft_note(st_row, r, expl, slot):
    return (f"Mentor note - {st_row['name']} ({st_row['id']})\nRisk category: {r['cat']} ({r['score']}/100)\n\nReasons:\n{expl}\n\n"
            "Suggested support actions:\n" + "\n".join("- " + x for x in recs(r)) + f"\n\nProposed meeting: {slot}")
def run_analysis(wk, use_llm=False, notes=True):
    key, model = st.session_state.get("gkey") or os.environ.get("GROQ_API_KEY", ""), st.session_state.get("gmodel", "llama-3.3-70b-versatile")
    n = 0
    for _, s in df("SELECT * FROM students").iterrows():
        if df("SELECT 1 FROM attendance WHERE student_id=? AND week=?", (s.id, wk)).empty: continue
        r = compute_risk(s.id, wk); prev = df("SELECT category FROM risk_analysis WHERE student_id=? ORDER BY id DESC LIMIT 1", (s.id,))
        changed = prev.empty or prev.category[0] != r["cat"]; expl, src = rule_text(s["name"], r), "rule-based"
        if use_llm and key and changed and r["cat"] != "Low":
            try: expl, src = llm_text(s["name"], r, key, model), "LLM (Groq)"
            except Exception: pass
        run("INSERT INTO risk_analysis(student_id,week,score,category,factors,explanation,source,created) VALUES(?,?,?,?,?,?,?,?)",
            (s.id, wk, r["score"], r["cat"], json.dumps(dict(flags=r["flags"], parts=r["parts"])), expl, src, now()))
        if notes and r["cat"] != "Low" and df("SELECT 1 FROM mentor_notes WHERE student_id=? AND status='Pending'", (s.id,)).empty:
            run("INSERT INTO mentor_notes(student_id,text,proposed_slot,status,created) VALUES(?,?,?,?,?)", (s.id, draft_note(s, r, expl, next_slot()), next_slot(), "Pending", now()))
        n += 1
    return n

# ---------------- data ----------------
def ingest(d):
    """long format: student_id,name,department,semester,subject,week,attendance,marks,submitted,missed"""
    for _, x in d.iterrows():
        run("INSERT OR IGNORE INTO students VALUES(?,?,?,?,?)", (str(x.student_id), x["name"], x.department, int(x.semester), 2))
        run("INSERT OR IGNORE INTO subjects(name) VALUES(?)", (x.subject,)); sid = df("SELECT id FROM subjects WHERE name=?", (x.subject,)).id[0]
        k = (str(x.student_id), int(sid), int(x.week))
        run("INSERT INTO attendance(student_id,subject_id,week,pct) VALUES(?,?,?,?)", k + (float(x.attendance),))
        run("INSERT INTO internal_marks(student_id,subject_id,week,marks) VALUES(?,?,?,?)", k + (float(x.marks),))
        run("INSERT INTO assignments(student_id,subject_id,week,submitted,missed) VALUES(?,?,?,?,?)", k + (int(x.submitted), int(x.missed)))
    w = df("SELECT a.student_id,a.week,(AVG(a.pct)+AVG(m.marks))/2 score FROM attendance a JOIN internal_marks m ON m.student_id=a.student_id AND m.week=a.week AND m.subject_id=a.subject_id GROUP BY a.student_id,a.week")
    run("DELETE FROM weekly_performance")
    for _, x in w.iterrows(): run("INSERT INTO weekly_performance(student_id,week,score) VALUES(?,?,?)", (x.student_id, int(x.week), float(x.score)))
def synth(n=60, weeks=8):
    rng = np.random.default_rng(7); fn = "Aarav Rahul Priya Ananya Vikram Sneha Karthik Divya Rohan Meera Arjun Kavya Nikhil Pooja Suresh Isha Manoj Neha Rakesh Lakshmi".split()
    ln = "Sharma Reddy Nair Iyer Gupta Rao Patel Kumar Singh Das".split(); depts = ["CSE", "ECE", "MECH", "CIVIL"]
    pool = ["Mathematics", "Data Structures", "Physics", "Electronics", "Thermodynamics", "Programming"]; rows = []
    for i in range(n):
        kind = rng.choice(["good", "avg", "decline"], p=[.5, .25, .25]); d = depts[i % 4]
        for sub in rng.choice(pool, 3, replace=False):
            a0, m0 = {"good": (90, 75), "avg": (76, 55), "decline": (84, 66)}[kind]; sl = {"good": 0, "avg": -.5, "decline": -3.2}[kind]
            for w in range(1, weeks + 1):
                pm = {"good": .95, "avg": .75, "decline": max(.2, .95 - .1 * w)}[kind]; sb = int(rng.random() < pm) + int(rng.random() < pm)
                rows.append(dict(student_id=f"S{1001 + i}", name=f"{fn[i % 20]} {ln[(i * 3) % 10]}", department=d, semester=3 + i % 4, subject=sub, week=w,
                                 attendance=float(np.clip(a0 + sl * w + rng.normal(0, 2), 30, 100)), marks=float(np.clip(m0 + sl * 2.2 * w + rng.normal(0, 4), 10, 100)), submitted=sb, missed=2 - sb))
    return pd.DataFrame(rows)
def init():
    if os.path.exists(DB): return
    with sqlite3.connect(DB) as c: c.executescript(SCHEMA)
    for u in [("admin", "admin123", "Admin", "Administrator"), ("mentor1", "mentor123", "Mentor", "Dr. Mehta"), ("mentor2", "mentor123", "Mentor", "Prof. Kapoor")]:
        run("INSERT INTO users(username,password_hash,role,name) VALUES(?,?,?,?)", (u[0], H(u[1]), u[2], u[3]))
    ingest(synth()); run("UPDATE students SET mentor_id=3 WHERE CAST(SUBSTR(id,2) AS INTEGER)%2=0")
    for w in range(4, 9): run_analysis(w, notes=(w == 8))

# ---------------- UI ----------------
st.set_page_config("EarlyAlert", "🎓", layout="wide"); init()
st.markdown("<style>.stApp{background:#f5f9ff}[data-testid=stSidebar]{background:#e8f1fd}h1,h2,h3{color:#0b4f9c}[data-testid=stMetric]{background:#fff;border:1px solid #cfe0f7;border-radius:12px;padding:12px}</style>", unsafe_allow_html=True)
COL = {"Low": "#2e9e5b", "Medium": "#f0a500", "High": "#d33f3f"}
U = st.session_state.get("user")
if not U:
    st.title("🎓 EarlyAlert – Academic Risk Early-Warning Agent")
    u, p = st.text_input("Username"), st.text_input("Password", type="password")
    if st.button("Login", type="primary"):
        r = df("SELECT * FROM users WHERE username=? AND password_hash=?", (u, H(p)))
        if r.empty: st.error("Invalid credentials")
        else: st.session_state.user = r.iloc[0].to_dict(); st.rerun()
    st.info("Demo logins — Admin: admin / admin123 · Mentor: mentor1 / mentor123 (or mentor2 / mentor123)"); st.stop()

admin = U["role"] == "Admin"
S = df("SELECT * FROM students"); S = S if admin else S[S.mentor_id == U["id"]]
ids = tuple(S.id) or ("",); ph = ",".join("?" * len(ids))
LR = df(f"SELECT r.* FROM risk_analysis r JOIN (SELECT MAX(id) m FROM risk_analysis GROUP BY student_id) x ON r.id=x.m WHERE student_id IN ({ph})", ids).merge(S[["id", "name", "department"]], left_on="student_id", right_on="id")
page = st.sidebar.radio("Navigate", ["Dashboard", "Student Management", "Student Profile", "Risk Analysis", "Mentor Recommendations", "Approval Center", "Weekly Reports", "Settings"])
st.sidebar.write(f"👤 {U['name']} ({U['role']})")
if st.sidebar.button("Logout"): st.session_state.pop("user"); st.rerun()
A = df(f"SELECT a.*,s.name sub FROM attendance a JOIN subjects s ON s.id=subject_id WHERE student_id IN ({ph})", ids)
M = df(f"SELECT a.*,s.name sub FROM internal_marks a JOIN subjects s ON s.id=subject_id WHERE student_id IN ({ph})", ids)
G = df(f"SELECT * FROM assignments WHERE student_id IN ({ph})", ids)
W = df(f"SELECT * FROM weekly_performance WHERE student_id IN ({ph})", ids)
pick = lambda: st.selectbox("Student", S.id, format_func=lambda i: f"{i} – {S.set_index('id').name[i]}")

if page == "Dashboard":
    st.title("Dashboard"); vc = LR.category.value_counts()
    c = st.columns(7); lw = A.week.max()
    for col, (l, v) in zip(c, [("Students", len(S)), ("High risk", vc.get("High", 0)), ("Medium", vc.get("Medium", 0)), ("Low", vc.get("Low", 0)),
                              ("Avg attendance", f"{A[A.week == lw].pct.mean():.1f}%"), ("Avg internal marks", f"{M[M.week == lw].marks.mean():.1f}"), ("Submission rate", f"{G.submitted.sum() / max(G.submitted.sum() + G.missed.sum(), 1):.0%}")]): col.metric(l, v)
    a, b = st.columns(2)
    a.plotly_chart(px.pie(LR, names="category", color="category", color_discrete_map=COL, title="Risk distribution"), use_container_width=True)
    b.plotly_chart(px.line(A.groupby("week").pct.mean().reset_index(), x="week", y="pct", markers=True, title="Average attendance trend"), use_container_width=True)
    a, b = st.columns(2)
    mm = M[M.week == lw].merge(S[["id", "department"]], left_on="student_id", right_on="id").groupby(["department", "sub"]).marks.mean().reset_index()
    a.plotly_chart(px.bar(mm, x="department", y="marks", color="sub", barmode="group", title="Internal marks (latest week) by department"), use_container_width=True)
    gg = G.merge(S[["id", "department"]], left_on="student_id", right_on="id").groupby("department")[["submitted", "missed"]].sum().reset_index()
    b.plotly_chart(px.bar(gg, x="department", y=["submitted", "missed"], title="Assignment submissions"), use_container_width=True)
    st.plotly_chart(px.line(W.groupby("week").score.mean().reset_index(), x="week", y="score", markers=True, title="Weekly academic performance"), use_container_width=True)

elif page == "Student Management":
    st.title("Student Management")
    t = S.merge(A[A.week == lw].groupby("student_id").pct.mean().rename("attendance %") if (lw := A.week.max()) else S, left_on="id", right_index=True, how="left")
    t = t.merge(M[M.week == lw].groupby("student_id").marks.agg(["mean", "sum"]).rename(columns={"mean": "internal avg", "sum": "total marks"}), left_on="id", right_index=True, how="left")
    t = t.merge(G.groupby("student_id")[["submitted", "missed"]].sum(), left_on="id", right_index=True, how="left")
    st.dataframe(t.round(1), use_container_width=True)
    if admin:
        st.subheader("Upload CSV"); st.caption("Columns: student_id,name,department,semester,subject,week,attendance,marks,submitted,missed")
        f = st.file_uploader("CSV", type="csv")
        if f and st.button("Import"):
            d = pd.read_csv(f); need = "student_id name department semester subject week attendance marks submitted missed".split()
            if [c for c in need if c not in d.columns]: st.error("Missing columns: " + ", ".join(c for c in need if c not in d.columns))
            else: ingest(d.dropna(subset=need)); st.success("Imported. Run weekly analysis to update risk.")

elif page == "Student Profile":
    st.title("Student Profile"); sid = pick(); s = S.set_index("id").loc[sid]; st.write(f"**{s['name']}** · {s.department} · Semester {s.semester}")
    r = LR[LR.student_id == sid].iloc[0]; st.metric("Risk score", r.score, r.category); st.info(r.explanation); st.caption(f"Explanation source: {r.source}")
    for c, (d, y, t) in zip(st.columns(2) * 2, [(A, "pct", "Attendance by subject"), (M, "marks", "Internal marks by subject")]): pass
    a, b = st.columns(2)
    a.plotly_chart(px.line(A[A.student_id == sid], x="week", y="pct", color="sub", markers=True, title="Attendance history"), use_container_width=True)
    b.plotly_chart(px.line(M[M.student_id == sid], x="week", y="marks", color="sub", markers=True, title="Internal marks history"), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(G[G.student_id == sid].groupby("week")[["submitted", "missed"]].sum().reset_index(), x="week", y=["submitted", "missed"], title="Assignments"), use_container_width=True)
    b.plotly_chart(px.line(W[W.student_id == sid], x="week", y="score", markers=True, title="Weekly performance"), use_container_width=True)
    h = df("SELECT week,score FROM risk_analysis WHERE student_id=? ORDER BY id", (sid,)); st.plotly_chart(px.line(h, x="week", y="score", markers=True, title="Risk score history"), use_container_width=True)
    st.subheader("Suggested mentor actions"); r2 = compute_risk(sid, int(A.week.max())); [st.write("• " + x) for x in recs(r2)]

elif page == "Risk Analysis":
    st.title("Risk Analysis"); st.caption("Score = 40% attendance risk + 35% marks risk + 25% assignment risk + trend adjustments (max +20). Low 0–39 · Medium 40–69 · High 70–100.")
    f = st.multiselect("Category", ["High", "Medium", "Low"], ["High", "Medium"]); t = LR[LR.category.isin(f)].sort_values("score", ascending=False)
    for _, x in t.iterrows():
        p = json.loads(x.factors)
        with st.expander(f"{x['name']} ({x.student_id}) — {x.category} · {x.score}"):
            st.write(x.explanation); st.json(p["parts"]); st.caption(f"Source: {x.source} · {x.created}")

elif page == "Mentor Recommendations":
    st.title("Mentor Recommendations")
    for _, x in LR[LR.category != "Low"].sort_values("score", ascending=False).iterrows():
        r = compute_risk(x.student_id, int(A.week.max()))
        with st.expander(f"{x['name']} — {x.category}"): [st.write("• " + y) for y in recs(r)]

elif page == "Approval Center":
    st.title("Approval Center"); st.warning("Nothing is sent automatically. Notes are only marked eligible after you approve them; this demo sends in-app only.")
    N = df(f"SELECT n.*,s.name FROM mentor_notes n JOIN students s ON s.id=student_id WHERE student_id IN ({ph}) ORDER BY n.id DESC", ids)
    for _, n in N[N.status == "Pending"].iterrows():
        st.subheader(f"{n['name']} ({n.student_id})"); txt = st.text_area("Edit note", n.text, height=220, key=f"t{n.id}"); c = st.columns(2)
        for lab, new, col in [("✅ Approve", "Approved", c[0]), ("❌ Reject", "Rejected", c[1])]:
            if col.button(lab, key=f"{new}{n.id}"):
                run("UPDATE mentor_notes SET text=?,status=? WHERE id=?", (txt, new, int(n.id))); run("INSERT INTO approval_history(note_id,action,by_user,at) VALUES(?,?,?,?)", (int(n.id), new, U["name"], now())); st.rerun()
    for _, n in N[N.status == "Approved"].iterrows():
        if st.button(f"Send in-app notification: {n['name']}", key=f"s{n.id}"):
            run("UPDATE mentor_notes SET status='Sent (in-app)' WHERE id=?", (int(n.id),)); run("INSERT INTO approval_history(note_id,action,by_user,at) VALUES(?,?,?,?)", (int(n.id), "Sent in-app", U["name"], now())); st.rerun()
    st.subheader("Approval history"); st.dataframe(df("SELECT h.at,h.action,h.by_user,n.student_id FROM approval_history h JOIN mentor_notes n ON n.id=note_id ORDER BY h.id DESC"), use_container_width=True)

elif page == "Weekly Reports":
    st.title("Weekly Reports"); st.caption("Analysis runs only when triggered below. No background scheduler is configured.")
    cw = int(df("SELECT MAX(week) w FROM attendance").w[0])
    if admin and st.button(f"Run weekly analysis for week {cw}", type="primary"):
        st.success(f"Analysed {run_analysis(cw, use_llm=True)} students; history kept.")
    H2 = df(f"SELECT student_id,week,score,category FROM risk_analysis WHERE student_id IN ({ph}) ORDER BY id", ids).drop_duplicates(["student_id", "week"], keep="last")
    st.plotly_chart(px.line(H2.groupby("week").score.mean().reset_index(), x="week", y="score", markers=True, title="Average risk score by week"), use_container_width=True)
    pv = H2.pivot(index="student_id", columns="week", values="score"); c = pv.columns
    if len(c) > 1:
        pv["change"] = pv[c[-1]] - pv[c[-2]]; st.subheader("Biggest increases vs previous week"); st.dataframe(pv.sort_values("change", ascending=False).head(10).round(1), use_container_width=True)

else:
    st.title("Settings"); st.text_input("Groq API key (session only; or set GROQ_API_KEY)", type="password", key="gkey")
    st.text_input("Model", "llama-3.3-70b-versatile", key="gmodel"); st.write("LLM:", "configured" if st.session_state.get("gkey") or os.environ.get("GROQ_API_KEY") else "not configured – rule-based explanations in use")
    st.write("External messaging: not configured (in-app notifications only).")
    if admin and st.button("Reset demo database"): os.remove(DB); st.session_state.clear(); st.rerun()