import { useState, useEffect } from "react";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from "recharts";

const API = "http://127.0.0.1:5000/api";

const C = {
  bg:      "#0b0f1a",
  surface: "#111827",
  card:    "#161d2e",
  border:  "#1e2d45",
  accent:  "#38bdf8",
  green:   "#34d399",
  red:     "#f87171",
  yellow:  "#fbbf24",
  purple:  "#a78bfa",
  muted:   "#64748b",
  text:    "#e2e8f0",
};

const CATEGORIES = ["Clothing","Electronics","Furniture","Groceries","Toys"];
const STORES     = ["S001","S002","S003","S004","S005"];

function KPICard({ label, value, icon, color, sub }) {
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderTop: `3px solid ${color || C.accent}`,
      borderRadius: 12, padding: "20px 24px", flex: 1, minWidth: 150,
    }}>
      <div style={{ fontSize: 22, marginBottom: 6 }}>{icon}</div>
      <div style={{ color: C.muted, fontSize: 11, letterSpacing: 1, textTransform: "uppercase" }}>{label}</div>
      <div style={{ color: color || C.accent, fontSize: 24, fontWeight: 700, fontFamily: "monospace", marginTop: 4 }}>{value}</div>
      {sub && <div style={{ color: C.muted, fontSize: 11, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{ color: C.text, fontSize: 13, fontWeight: 700, letterSpacing: 2,
      textTransform: "uppercase", marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ width: 3, height: 14, background: C.accent, borderRadius: 2, display: "inline-block" }} />
      {children}
    </div>
  );
}

const fmt     = n => n == null ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
const fmtM    = n => n == null ? "—" : `$${(Number(n)/1000000).toFixed(1)}M`;
const fmtDate = d => d ? d.slice(5) : ""; // show MM-DD only

// ── OVERVIEW PAGE ─────────────────────────────────────────────
function OverviewPage({ category, store }) {
  const [kpis,  setKpis]  = useState(null);
  const [sales, setSales] = useState([]);
  const [stats, setStats] = useState([]);

  useEffect(() => {
    fetch(`${API}/dashboard-summary`).then(r=>r.json()).then(setKpis).catch(()=>{});
    fetch(`${API}/category-stats`).then(r=>r.json()).then(setStats).catch(()=>{});
  }, []);

  useEffect(() => {
    fetch(`${API}/sales/${category}?days=60`)
      .then(r=>r.json()).then(setSales).catch(()=>{});
  }, [category]);

  return (
    <div>
      {/* KPI Row */}
      <div style={{ display:"flex", gap:14, flexWrap:"wrap", marginBottom:24 }}>
        <KPICard icon="💰" label="Total Revenue"   value={fmtM(kpis?.total_revenue)}   color={C.accent} />
        <KPICard icon="📦" label="Total Units Sold" value={fmt(kpis?.total_units)}       color={C.green}  />
        <KPICard icon="🏪" label="Stores"           value={kpis?.total_stores || "—"}   color={C.purple} />
        <KPICard icon="🎯" label="Base MAPE"        value={`${kpis?.base_mape || "—"}%`} color={C.yellow} sub="dataset forecast" />
        <KPICard icon="🧠" label="LSTM MAPE"        value={`${kpis?.lstm_mape || "—"}%`} color={C.green}  sub="our model" />
      </div>

      {/* Revenue Chart */}
      <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:12, padding:24, marginBottom:20 }}>
        <SectionTitle>Daily Revenue — {category} (Last 60 Days)</SectionTitle>
        {sales.length === 0 ? (
          <div style={{ color:C.muted, textAlign:"center", padding:40 }}>Loading chart data...</div>
        ) : (
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={sales}>
              <defs>
                <linearGradient id="rev" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.accent} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={C.accent} stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="fc" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.green} stopOpacity={0.2}/>
                  <stop offset="95%" stopColor={C.green} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="date" tick={{fill:C.muted,fontSize:10}} tickLine={false}
                tickFormatter={fmtDate} interval={9} />
              <YAxis tick={{fill:C.muted,fontSize:10}} tickLine={false} axisLine={false}
                tickFormatter={v=>`$${(v/1000).toFixed(0)}k`} />
              <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}}
                labelStyle={{color:C.text}} formatter={(v,n)=>[`$${fmt(v)}`,n]} />
              <Legend wrapperStyle={{color:C.muted,fontSize:12}} />
              <Area type="monotone" dataKey="revenue"  stroke={C.accent} fill="url(#rev)" strokeWidth={2} name="Revenue"  dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Units + Forecast Chart */}
      <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:12, padding:24, marginBottom:20 }}>
        <SectionTitle>Units Sold vs Forecast — {category}</SectionTitle>
        {sales.length === 0 ? (
          <div style={{ color:C.muted, textAlign:"center", padding:40 }}>Loading...</div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={sales.slice(-30)}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="date" tick={{fill:C.muted,fontSize:10}} tickLine={false}
                tickFormatter={fmtDate} interval={6} />
              <YAxis tick={{fill:C.muted,fontSize:10}} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}} />
              <Legend wrapperStyle={{color:C.muted,fontSize:12}} />
              <Line type="monotone" dataKey="units"    stroke={C.accent} strokeWidth={2} name="Actual Units"   dot={false} />
              <Line type="monotone" dataKey="forecast" stroke={C.green}  strokeWidth={2} name="Base Forecast"  dot={false} strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Category Comparison */}
      {stats.length > 0 && (
        <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:12, padding:24 }}>
          <SectionTitle>MAPE Comparison — All Categories</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={stats}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="category" tick={{fill:C.muted,fontSize:11}} tickLine={false} />
              <YAxis tick={{fill:C.muted,fontSize:10}} tickLine={false} axisLine={false}
                tickFormatter={v=>`${v}%`} />
              <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}}
                formatter={(v)=>[`${v}%`]} />
              <Legend wrapperStyle={{color:C.muted,fontSize:12}} />
              <Bar dataKey="base_mape" fill={C.yellow} radius={[3,3,0,0]} name="Base MAPE %" />
              <Bar dataKey="lstm_mape" fill={C.green}  radius={[3,3,0,0]} name="LSTM MAPE %" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ── FORECAST PAGE ─────────────────────────────────────────────
function ForecastPage({ category, store }) {
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetch(`${API}/sales/${category}?days=30`)
      .then(r=>r.json()).then(setHistory).catch(()=>{});
  }, [category]);

  const runForecast = async () => {
    setLoading(true); setResult(null);
    try {
      const res  = await fetch(`${API}/predict`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ category, store })
      });
      setResult(await res.json());
    } catch(e) { setResult({ error:"Could not connect to API" }); }
    setLoading(false);
  };

  return (
    <div>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:20 }}>
        <SectionTitle>Demand Forecast — {category} / {store}</SectionTitle>
        <button onClick={runForecast} disabled={loading} style={{
          background:C.accent, color:"#000", border:"none", borderRadius:8,
          padding:"10px 24px", fontWeight:700, cursor:loading?"not-allowed":"pointer",
          fontSize:13, opacity:loading?0.6:1,
        }}>{loading?"⏳ Running...":"▶ Run Forecast"}</button>
      </div>

      {result && (
        <div style={{
          background: result.error ? `${C.red}22` : `${C.green}22`,
          border:`1px solid ${result.error ? C.red : C.green}`,
          borderRadius:10, padding:20, marginBottom:20,
        }}>
          {result.error ? (
            <div style={{color:C.red}}>{result.error}</div>
          ) : (
            <div>
              <div style={{color:C.green,fontWeight:700,fontSize:15,marginBottom:12}}>
                ✅ Forecast Generated for {result.forecast_date}
              </div>
              <div style={{display:"flex",gap:32,flexWrap:"wrap"}}>
                <div>
                  <div style={{color:C.muted,fontSize:11,textTransform:"uppercase"}}>Base Forecast</div>
                  <div style={{color:C.yellow,fontSize:22,fontWeight:700,fontFamily:"monospace"}}>
                    {Number(result.base_forecast).toFixed(0)} units
                  </div>
                </div>
                <div>
                  <div style={{color:C.muted,fontSize:11,textTransform:"uppercase"}}>LSTM Correction</div>
                  <div style={{color:result.lstm_correction>=0?C.green:C.red,fontSize:22,fontWeight:700,fontFamily:"monospace"}}>
                    {result.lstm_correction>=0?"+":""}{Number(result.lstm_correction).toFixed(1)} units
                  </div>
                </div>
                <div>
                  <div style={{color:C.muted,fontSize:11,textTransform:"uppercase"}}>Final Prediction</div>
                  <div style={{color:C.accent,fontSize:26,fontWeight:700,fontFamily:"monospace"}}>
                    {Number(result.predicted_units).toFixed(0)} units
                  </div>
                </div>
                <div>
                  <div style={{color:C.muted,fontSize:11,textTransform:"uppercase"}}>Confidence Range</div>
                  <div style={{color:C.text,fontSize:16,fontFamily:"monospace"}}>
                    {Number(result.confidence_low).toFixed(0)} — {Number(result.confidence_high).toFixed(0)}
                  </div>
                </div>
              </div>
              <div style={{color:C.muted,fontSize:12,marginTop:10}}>
                🎯 LSTM improved accuracy by ~{result.mape_improvement} over base forecast
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recent sales as context */}
      {history.length > 0 && (
        <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:24}}>
          <SectionTitle>Recent Sales Context — {category}</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="u" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.accent} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={C.accent} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="date" tick={{fill:C.muted,fontSize:10}} tickLine={false} tickFormatter={fmtDate} />
              <YAxis tick={{fill:C.muted,fontSize:10}} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}} />
              <Area type="monotone" dataKey="units" stroke={C.accent} fill="url(#u)" strokeWidth={2} name="Units Sold" dot={false} />
              <Line type="monotone" dataKey="forecast" stroke={C.green} strokeWidth={2} name="Base Forecast" dot={false} strokeDasharray="4 4" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ── ANOMALY PAGE ──────────────────────────────────────────────
function AnomalyPage({ category, store }) {
  const [anomalies, setAnomalies] = useState([]);
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);

  useEffect(() => {
    fetch(`${API}/anomalies`).then(r=>r.json()).then(setAnomalies).catch(()=>{});
  }, []);

  const runDetection = async () => {
    setLoading(true); setResult(null);
    try {
      const res  = await fetch(`${API}/detect-anomaly`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ category, store })
      });
      setResult(await res.json());
      fetch(`${API}/anomalies`).then(r=>r.json()).then(setAnomalies).catch(()=>{});
    } catch(e) { setResult({ error:"Could not connect to API" }); }
    setLoading(false);
  };

  const sevColor = s => s==="high"?C.red:s==="medium"?C.yellow:C.green;

  return (
    <div>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:20}}>
        <SectionTitle>Anomaly Detection — {category} / {store}</SectionTitle>
        <button onClick={runDetection} disabled={loading} style={{
          background:C.red,color:"#fff",border:"none",borderRadius:8,
          padding:"10px 24px",fontWeight:700,cursor:loading?"not-allowed":"pointer",
          fontSize:13,opacity:loading?0.6:1,
        }}>{loading?"⏳ Detecting...":"🔍 Run Detection"}</button>
      </div>

      {result && (
        <div style={{
          background:result.is_anomaly?`${C.red}22`:`${C.green}22`,
          border:`1px solid ${result.is_anomaly?C.red:C.green}`,
          borderRadius:10,padding:20,marginBottom:20,
        }}>
          {result.error ? (
            <div style={{color:C.red}}>{result.error}</div>
          ) : result.is_anomaly ? (
            <div>
              <div style={{color:C.red,fontWeight:700,fontSize:15}}>
                🚨 ANOMALY DETECTED in {result.category}!
              </div>
              <div style={{color:C.muted,fontSize:13,marginTop:8}}>
                Type: <b style={{color:C.text}}>{result.type?.toUpperCase()}</b> &nbsp;|&nbsp;
                Severity: <b style={{color:sevColor(result.severity)}}>{result.severity?.toUpperCase()}</b> &nbsp;|&nbsp;
                Recon. Error: <b style={{color:C.text,fontFamily:"monospace"}}>{result.recon_error}</b> &nbsp;|&nbsp;
                Threshold: <b style={{color:C.text,fontFamily:"monospace"}}>{result.threshold}</b>
              </div>
            </div>
          ) : (
            <div style={{color:C.green,fontWeight:700}}>
              ✅ No anomaly detected in {result.category} — sales within normal range
            </div>
          )}
        </div>
      )}

      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,overflow:"hidden"}}>
        <div style={{padding:"16px 20px",borderBottom:`1px solid ${C.border}`}}>
          <SectionTitle>Detected Anomalies Log</SectionTitle>
        </div>
        {anomalies.length===0 ? (
          <div style={{padding:40,textAlign:"center",color:C.muted}}>✅ No anomalies found!</div>
        ) : (
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
            <thead>
              <tr style={{background:C.surface}}>
                {["Category","Date","Revenue","Type","Severity","Error"].map(h=>(
                  <th key={h} style={{padding:"12px 16px",textAlign:"left",color:C.muted,
                    fontWeight:600,borderBottom:`1px solid ${C.border}`}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {anomalies.map((a,i)=>(
                <tr key={i} style={{borderBottom:`1px solid ${C.border}`}}>
                  <td style={{padding:"11px 16px",color:C.text}}>{a.category}</td>
                  <td style={{padding:"11px 16px",color:C.muted}}>{a.date}</td>
                  <td style={{padding:"11px 16px",color:C.text,fontFamily:"monospace"}}>${Number(a.revenue||0).toFixed(2)}</td>
                  <td style={{padding:"11px 16px"}}>
                    <span style={{background:`${a.type==="spike"?C.red:C.yellow}22`,
                      color:a.type==="spike"?C.red:C.yellow,
                      padding:"2px 10px",borderRadius:20,fontSize:11,fontWeight:700}}>
                      {a.type?.toUpperCase()}
                    </span>
                  </td>
                  <td style={{padding:"11px 16px",color:sevColor(a.severity),fontWeight:700}}>{a.severity}</td>
                  <td style={{padding:"11px 16px",color:C.muted,fontFamily:"monospace"}}>{Number(a.recon_err||0).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ── INVENTORY PAGE ────────────────────────────────────────────
function InventoryPage() {
  const [inventory, setInventory] = useState([]);

  useEffect(()=>{
    fetch(`${API}/inventory`).then(r=>r.json()).then(setInventory).catch(()=>{});
  },[]);

  return (
    <div>
      <SectionTitle>Inventory Insights & Recommendations</SectionTitle>
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))",gap:16}}>
        {inventory.map((item,i)=>(
          <div key={i} style={{
            background:C.card,
            border:`1px solid ${item.reorder_flag?C.red:item.overstock_flag?C.yellow:C.border}`,
            borderRadius:12,padding:20,
          }}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:12}}>
              <div style={{color:C.text,fontWeight:700,fontSize:15}}>{item.category}</div>
              <div style={{color:C.muted,fontSize:12}}>${Number(item.avg_daily_revenue).toFixed(0)}/day</div>
            </div>
            <div style={{display:"flex",gap:20,marginBottom:14}}>
              <div>
                <div style={{color:C.muted,fontSize:10,textTransform:"uppercase",marginBottom:2}}>Avg Stock</div>
                <div style={{color:C.text,fontWeight:700,fontFamily:"monospace",fontSize:18}}>{Number(item.current_stock).toFixed(0)}</div>
              </div>
              <div>
                <div style={{color:C.muted,fontSize:10,textTransform:"uppercase",marginBottom:2}}>Recommended</div>
                <div style={{color:C.accent,fontWeight:700,fontFamily:"monospace",fontSize:18}}>{Number(item.recommended_stock).toFixed(0)}</div>
              </div>
              <div>
                <div style={{color:C.muted,fontSize:10,textTransform:"uppercase",marginBottom:2}}>Avg Daily</div>
                <div style={{color:C.green,fontWeight:700,fontFamily:"monospace",fontSize:18}}>{Number(item.avg_daily_units).toFixed(0)}</div>
              </div>
            </div>
            {item.reorder_flag?(
              <div style={{background:`${C.red}22`,color:C.red,borderRadius:6,padding:"8px 12px",fontSize:12,fontWeight:600}}>
                🔴 Reorder Required — Stock Below Safe Level
              </div>
            ):item.overstock_flag?(
              <div style={{background:`${C.yellow}22`,color:C.yellow,borderRadius:6,padding:"8px 12px",fontSize:12,fontWeight:600}}>
                🟡 Overstocked — Consider Markdown
              </div>
            ):(
              <div style={{background:`${C.green}22`,color:C.green,borderRadius:6,padding:"8px 12px",fontSize:12,fontWeight:600}}>
                ✅ Stock Level OK
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── ROOT APP ──────────────────────────────────────────────────
export default function InventoryApp() {
  const [page,     setPage]     = useState("overview");
  const [category, setCategory] = useState("Clothing");
  const [store,    setStore]    = useState("S001");

  const navItems = [
    { id:"overview",  label:"📊 Overview"  },
    { id:"forecast",  label:"🔮 Forecast"  },
    { id:"anomalies", label:"⚠️ Anomalies" },
    { id:"inventory", label:"📦 Inventory" },
  ];

  return (
    <div style={{background:C.bg,minHeight:"100vh",color:C.text,fontFamily:"'Segoe UI',system-ui,sans-serif"}}>
      {/* TOP NAV */}
      <div style={{background:C.surface,borderBottom:`1px solid ${C.border}`,
        padding:"0 28px",display:"flex",alignItems:"center",justifyContent:"space-between",height:58}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <span style={{fontSize:22}}>🧠</span>
          <div>
            <div style={{fontWeight:700,fontSize:14,color:C.text}}>Smart Inventory AI</div>
            <div style={{color:C.muted,fontSize:10}}>Retail Store · Deep Learning · LSTM</div>
          </div>
        </div>
        <div style={{display:"flex",gap:4}}>
          {navItems.map(n=>(
            <button key={n.id} onClick={()=>setPage(n.id)} style={{
              background:page===n.id?`${C.accent}22`:"transparent",
              color:page===n.id?C.accent:C.muted,
              border:page===n.id?`1px solid ${C.accent}44`:"1px solid transparent",
              borderRadius:8,padding:"6px 16px",cursor:"pointer",
              fontSize:13,fontWeight:page===n.id?600:400,
            }}>{n.label}</button>
          ))}
        </div>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{display:"flex",alignItems:"center",gap:6}}>
            <label style={{color:C.muted,fontSize:12}}>Category</label>
            <select value={category} onChange={e=>setCategory(e.target.value)} style={{
              background:C.card,color:C.text,border:`1px solid ${C.border}`,
              borderRadius:6,padding:"5px 10px",fontSize:13
            }}>
              {CATEGORIES.map(c=><option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:6}}>
            <label style={{color:C.muted,fontSize:12}}>Store</label>
            <select value={store} onChange={e=>setStore(e.target.value)} style={{
              background:C.card,color:C.text,border:`1px solid ${C.border}`,
              borderRadius:6,padding:"5px 10px",fontSize:13
            }}>
              {STORES.map(s=><option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* CONTENT */}
      <div style={{maxWidth:1200,margin:"0 auto",padding:"28px 24px"}}>
        {page==="overview"  && <OverviewPage  category={category} store={store} />}
        {page==="forecast"  && <ForecastPage  category={category} store={store} />}
        {page==="anomalies" && <AnomalyPage   category={category} store={store} />}
        {page==="inventory" && <InventoryPage />}
      </div>

      <div style={{textAlign:"center",color:C.muted,fontSize:11,paddingBottom:20,letterSpacing:1}}>
        SMART INVENTORY AI · LSTM RESIDUAL LEARNING · AUTOENCODER ANOMALY DETECTION · MYSQL
      </div>
    </div>
  );
}