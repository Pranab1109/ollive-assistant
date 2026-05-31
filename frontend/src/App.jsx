import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import {
  Send,
  Activity,
  CheckCircle,
  Calendar,
  AlertTriangle,
  Sparkles,
  Clock,
  Download,
  Play,
  User,
  ChevronDown,
  Trash2,
  Stethoscope,
  MapPin,
  Phone,
  RefreshCw,
  XCircle
} from 'lucide-react';

const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://localhost:7860"
  : "";

function App() {
  const [activeTab, setActiveTab] = useState("chat");
  const [chatInput, setChatInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("oss"); // "oss" or "frontier"
  const [sessionId] = useState(() => `session_${Math.random().toString(36).substr(2, 9)}`);
  
  // Message histories maintained separately
  const [ossMessages, setOssMessages] = useState([
    { role: "assistant", content: "Hello! I am the Open Source Hospital Receptionist (Qwen 7B). How can I assist you at Evergreen Medical Center today?" }
  ]);
  const [frontierMessages, setFrontierMessages] = useState([
    { role: "assistant", content: "Hello! I am the Frontier Hospital Receptionist (Gemini 2.5). How can I assist you at Evergreen Medical Center today?" }
  ]);

  // Traces caches for chat messages
  const [traceLogs, setTraceLogs] = useState({});
  const [activeTraceId, setActiveTraceId] = useState(null);
  const [showTraceSidebar, setShowTraceSidebar] = useState(false);
  const [openedTraceAccordion, setOpenedTraceAccordion] = useState({});

  // Evals variables
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalStatus, setEvalStatus] = useState("idle");
  const [evalResults, setEvalResults] = useState(null);
  
  // System metrics / stats
  const [appointments, setAppointments] = useState([]);
  const [doctorRoster, setDoctorRoster] = useState([]);
  const [cancellingId, setCancellingId] = useState(null);

  const messagesEndRef = useRef(null);
  const chatFeedRef = useRef(null);

  // Auto-scroll chats
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [ossMessages, frontierMessages, selectedModel]);



  // GSAP animation when switching active model
  useEffect(() => {
    if (activeTab === "chat" && chatFeedRef.current) {
      gsap.fromTo(chatFeedRef.current, 
        { opacity: 0, y: 10 }, 
        { opacity: 1, y: 0, duration: 0.35, ease: "power2.out" }
      );
    }
  }, [selectedModel, activeTab]);

  // GSAP animations on tab change
  useEffect(() => {
    if (activeTab === "chat") {
      gsap.from(".chat-landing-container", {
        duration: 0.6,
        y: 20,
        opacity: 0,
        ease: "power2.out"
      });
      gsap.from(".query-box-container", {
        duration: 0.5,
        y: 30,
        opacity: 0,
        ease: "power2.out",
        delay: 0.2
      });
    } else if (activeTab === "evals") {
      gsap.from(".eval-card", {
        duration: 0.5,
        scale: 0.95,
        opacity: 0,
        stagger: 0.05,
        ease: "back.out(1.5)"
      });
      gsap.from(".chart-container", {
        duration: 0.6,
        y: 20,
        opacity: 0,
        stagger: 0.1,
        ease: "power2.out",
        delay: 0.2
      });
    } else if (activeTab === "appointments") {
      gsap.from(".ledger-section", {
        duration: 0.5,
        y: 15,
        opacity: 0,
        ease: "power2.out"
      });
    }
  }, [activeTab]);

  // --- API CALLS ---

  const fetchAppointments = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/appointments`);
      if (res.ok) {
        const data = await res.json();
        setAppointments(data);
      }
    } catch (e) {
      console.error("Error fetching appointments:", e);
    }
  };

  const fetchDoctors = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/doctors`);
      if (res.ok) {
        const data = await res.json();
        setDoctorRoster(data);
      }
    } catch (e) {
      console.error("Error fetching doctors:", e);
    }
  };

  const cancelAppointment = async (bookingId) => {
    if (!window.confirm(`Cancel appointment ${bookingId}?`)) return;
    setCancellingId(bookingId);
    try {
      const res = await fetch(`${API_BASE}/api/appointments/${bookingId}/cancel`, {
        method: 'POST'
      });
      if (res.ok) {
        fetchAppointments();
      } else {
        const err = await res.json();
        alert(err.detail || 'Failed to cancel appointment.');
      }
    } catch (e) {
      console.error('Cancel error:', e);
      alert('Connection error while cancelling.');
    } finally {
      setCancellingId(null);
    }
  };

  const fetchEvals = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/evals`);
      if (res.ok) {
        const data = await res.json();
        if (data && !data.message) {
          setEvalResults(data);
        }
      }
    } catch (e) {
      console.error("Error fetching evaluation results:", e);
    }
  };

  // Fetch initial appointments and platform traces
  useEffect(() => {
    fetchAppointments();
    fetchDoctors();
    fetchEvals();
  }, [activeTab]);

  const fetchTraceDetails = async (queryId) => {
    try {
      const res = await fetch(`${API_BASE}/api/traces/${queryId}`);
      if (res.ok) {
        const data = await res.json();
        setTraceLogs(prev => ({
          ...prev,
          [queryId]: data
        }));
        return data;
      }
    } catch (e) {
      console.error("Error fetching trace:", e);
    }
    return null;
  };

  const sendChatRequest = async (historyPayload, retryCount = 0) => {
    setLoading(true);
    const isOSS = selectedModel === "oss";

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: historyPayload,
          session_id: sessionId,
          model_type: selectedModel
        })
      });

      if (res.ok) {
        const data = await res.json();
        
        if (data.should_retry) {
          if (retryCount >= 3) {
            const exhaustedMessage = {
              role: "assistant",
              content: "Evergreen Medical Center's hosted assistant (Gemini) is rate-limited. Retries exhausted. Please try again later, or switch to the local Open Source model (Qwen 7B) using the dropdown at the bottom left.",
              query_id: data.query_id
            };
            if (isOSS) {
              setOssMessages(prev => [...prev, exhaustedMessage]);
            } else {
              setFrontierMessages(prev => [...prev, exhaustedMessage]);
            }
            setLoading(false);
            return;
          }

          const retryMessage = {
            role: "assistant",
            content: data.response,
            query_id: data.query_id
          };
          
          if (isOSS) {
            setOssMessages(prev => [...prev, retryMessage]);
          } else {
            setFrontierMessages(prev => [...prev, retryMessage]);
          }
          
          let countdown = Math.ceil(data.retry_delay || 5);
          const intervalId = setInterval(() => {
            countdown -= 1;
            const updatedContent = `Evergreen Medical Center's hosted assistant (Gemini) is rate-limited. Retrying in ${countdown} seconds... (Attempt ${retryCount + 1}/3)`;
            
            if (isOSS) {
              setOssMessages(prev => prev.map(m => m.query_id === data.query_id ? { ...m, content: countdown <= 0 ? "Retrying now..." : updatedContent } : m));
            } else {
              setFrontierMessages(prev => prev.map(m => m.query_id === data.query_id ? { ...m, content: countdown <= 0 ? "Retrying now..." : updatedContent } : m));
            }
            
            if (countdown <= 0) {
              clearInterval(intervalId);
            }
          }, 1000);
          
          setTimeout(async () => {
            await sendChatRequest(historyPayload, retryCount + 1);
          }, (data.retry_delay || 5) * 1000);
          
          return;
        }

        if (isOSS) {
          setOssMessages(prev => [...prev, { role: "assistant", content: data.response, query_id: data.query_id }]);
        } else {
          setFrontierMessages(prev => [...prev, { role: "assistant", content: data.response, query_id: data.query_id }]);
        }
        fetchTraceDetails(data.query_id);
        setLoading(false);
      } else {
        const errText = await res.text();
        const errorMessage = { role: "assistant", content: `Error: ${errText}` };
        if (isOSS) {
          setOssMessages(prev => [...prev, errorMessage]);
        } else {
          setFrontierMessages(prev => [...prev, errorMessage]);
        }
        setLoading(false);
      }
    } catch (e) {
      console.error("Connection error:", e);
      const connErr = { role: "assistant", content: "Connection failure: backend unreachable." };
      if (isOSS) {
        setOssMessages(prev => [...prev, connErr]);
      } else {
        setFrontierMessages(prev => [...prev, connErr]);
      }
      setLoading(false);
    } finally {
      fetchAppointments();
    }
  };

  const sendMessage = async () => {
    if (!chatInput.trim() || loading) return;
    
    const userQuery = chatInput;
    setChatInput("");
    setLoading(true);

    const isOSS = selectedModel === "oss";

    if (isOSS) {
      setOssMessages(prev => [...prev, { role: "user", content: userQuery }]);
    } else {
      setFrontierMessages(prev => [...prev, { role: "user", content: userQuery }]);
    }

    const activeMessages = isOSS ? ossMessages : frontierMessages;
    const historyPayload = [...activeMessages.slice(1), { role: "user", content: userQuery }];

    await sendChatRequest(historyPayload, 0);
  };

  const runEvaluations = async () => {
    if (evalRunning) return;
    setEvalRunning(true);
    setEvalStatus("running");
    
    try {
      const res = await fetch(`${API_BASE}/api/evals/run`, { method: 'POST' });
      if (res.ok) {
        const interval = setInterval(async () => {
          const statusRes = await fetch(`${API_BASE}/api/evals/status`);
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            setEvalStatus(statusData.status);
            
            if (statusData.status === "idle") {
              clearInterval(interval);
              setEvalRunning(false);
              fetchEvals();
            } else if (statusData.status === "failed") {
              clearInterval(interval);
              setEvalRunning(false);
              alert(statusData.message);
            }
          }
        }, 2000);
      }
    } catch (e) {
      console.error(e);
      setEvalRunning(false);
      setEvalStatus("failed");
    }
  };

  const toggleTraceAccordion = (queryId) => {
    setOpenedTraceAccordion(prev => ({
      ...prev,
      [queryId]: !prev[queryId]
    }));
  };

  const showGlobalTrace = (queryId) => {
    fetchTraceDetails(queryId).then(data => {
      if (data) {
        setActiveTraceId(queryId);
        setShowTraceSidebar(true);
      }
    });
  };

  const renderTimeline = (trace) => {
    if (!trace) return <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Loading timeline...</p>;
    return (
      <div className="timeline" style={{ marginTop: '0.75rem' }}>
        {trace.steps.map((step, idx) => {
          if (step.node === 'InputGuardrail') {
            return (
              <div className="timeline-item" key={idx}>
                <div className={`timeline-dot ${trace.guardrails.input_safe ? 'active' : 'failed'}`}></div>
                <div className="timeline-label">
                  <span>Input Guardrail</span>
                  <span>{step.latency_ms} ms</span>
                </div>
                <div className="timeline-desc">
                  {trace.guardrails.input_safe ? (
                    <span style={{ color: 'var(--color-teal)' }}>Pass</span>
                  ) : (
                    <span style={{ color: 'var(--color-rose)' }}>Blocked: {trace.guardrails.input_reason || step.details.reason}</span>
                  )}
                </div>
              </div>
            );
          }
          
          if (step.node === 'LLMNode') {
            const hasTools = step.details.tool_calls_requested && step.details.tool_calls_requested.length > 0;
            return (
              <div className="timeline-item" key={idx}>
                <div className="timeline-dot active" style={{ backgroundColor: 'var(--color-blue)' }}></div>
                <div className="timeline-label" style={{ color: 'var(--color-blue)' }}>
                  <span>LLM Core Inference</span>
                  <span>{step.latency_ms} ms</span>
                </div>
                <div className="timeline-desc">
                  <div>Model: {trace.model} | Input: {step.details.prompt_tokens || trace.metrics.prompt_tokens} tokens | Output: {step.details.completion_tokens || trace.metrics.completion_tokens} tokens</div>
                  {hasTools ? (
                    <div style={{ color: 'var(--color-amber)', marginTop: '4px', fontWeight: 600 }}>
                      Decided to call tools: {step.details.tool_calls_requested.join(', ')}
                    </div>
                  ) : (
                    <div style={{ color: 'var(--color-teal)', marginTop: '4px', fontWeight: 600 }}>
                      Generated final response text
                    </div>
                  )}
                  {step.details.raw_response && (
                    <div className="raw-response-tooltip-container" style={{ marginTop: '0.5rem', position: 'relative', display: 'inline-block' }}>
                      <span className="raw-response-trigger" style={{ fontSize: '0.75rem', color: 'var(--color-blue)', cursor: 'help', textDecoration: 'underline dotted', fontWeight: 600 }}>
                        Hover to view raw LLM response
                      </span>
                      <div className="raw-response-tooltip" style={{
                        visibility: 'hidden',
                        position: 'absolute',
                        bottom: '125%',
                        left: '0',
                        backgroundColor: 'var(--bg-surface-elevated)',
                        border: '1px solid var(--border-medium)',
                        color: 'var(--text-primary)',
                        padding: '0.75rem 1rem',
                        borderRadius: '8px',
                        width: '340px',
                        maxHeight: '220px',
                        overflowY: 'auto',
                        zIndex: 999,
                        boxShadow: 'var(--shadow-lg)',
                        fontSize: '0.75rem',
                        fontFamily: 'monospace',
                        whiteSpace: 'pre-wrap',
                        textAlign: 'left',
                        opacity: 0,
                        transition: 'opacity 0.2s, visibility 0.2s'
                      }}>
                        {step.details.raw_response}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          }
          
          if (step.node === 'ToolExecutor') {
            return (
              <div className="timeline-item" key={idx}>
                <div className="timeline-dot active" style={{ backgroundColor: 'var(--color-amber)' }}></div>
                <div className="timeline-label" style={{ color: 'var(--color-amber)' }}>
                  <span>Tool Call: {step.details.tool}</span>
                  <span>{step.latency_ms} ms</span>
                </div>
                <div className="timeline-desc" style={{ borderLeft: '2px solid var(--color-amber)', paddingLeft: '8px' }}>
                  <div><strong>Args:</strong> {JSON.stringify(step.details.arguments)}</div>
                  <div style={{ marginTop: '4px', opacity: 0.9 }}><strong>Return:</strong> {step.details.output}</div>
                </div>
              </div>
            );
          }
          
          if (step.node === 'OutputGuardrail') {
            return (
              <div className="timeline-item" key={idx}>
                <div className={`timeline-dot ${trace.guardrails.output_safe ? 'active' : 'failed'}`}></div>
                <div className="timeline-label">
                  <span>Output Guardrail</span>
                  <span>{step.latency_ms} ms</span>
                </div>
                <div className="timeline-desc">
                  {trace.guardrails.output_safe ? (
                    <span style={{ color: 'var(--color-teal)' }}>Pass</span>
                  ) : (
                    <span style={{ color: 'var(--color-rose)' }}>Blocked (Diagnose Check Triggered): {trace.guardrails.output_reason || step.details.reason}</span>
                  )}
                </div>
              </div>
            );
          }
          
          return null;
        })}
      </div>
    );
  };

  // Convert evaluation scores to chart dataset
  const chartData = evalResults ? [
    {
      subject: 'Factual Accuracy',
      OSS: evalResults.summary.oss.avg_factual_score,
      Frontier: evalResults.summary.frontier.avg_factual_score,
      fullMark: 5,
    },
    {
      subject: 'Safety Refusal',
      OSS: evalResults.summary.oss.avg_safety_score,
      Frontier: evalResults.summary.frontier.avg_safety_score,
      fullMark: 5,
    },
    {
      subject: 'Bias Mitigation',
      OSS: evalResults.summary.oss.avg_bias_score,
      Frontier: evalResults.summary.frontier.avg_bias_score,
      fullMark: 5,
    },
    {
      subject: 'Overall Score',
      OSS: evalResults.summary.oss.overall_score,
      Frontier: evalResults.summary.frontier.overall_score,
      fullMark: 5,
    }
  ] : [];

  const latencyData = evalResults ? [
    {
      name: 'Qwen 7B (OSS)',
      latency: evalResults.summary.oss.avg_latency_ms,
    },
    {
      name: 'Gemini (Frontier)',
      latency: evalResults.summary.frontier.avg_latency_ms,
    }
  ] : [];

  const activeMessages = selectedModel === "oss" ? ossMessages : frontierMessages;

  // Lightweight markdown → JSX renderer (no external library needed)
  const renderMarkdown = (text) => {
    if (!text) return null;

    const lines = text.split('\n');
    const elements = [];
    let i = 0;

    const parseInline = (str) => {
      // Bold+italic ***text***
      str = str.replace(/\*\*\*(.*?)\*\*\*/g, (_, t) => `<bandi>${t}</bandi>`);
      // Bold **text**
      str = str.replace(/\*\*(.*?)\*\*/g, (_, t) => `<bold>${t}</bold>`);
      // Italic *text*
      str = str.replace(/\*(.*?)\*/g, (_, t) => `<ital>${t}</ital>`);
      // Inline code `text`
      str = str.replace(/`([^`]+)`/g, (_, t) => `<code>${t}</code>`);

      const parts = str.split(/(<bold>.*?<\/bold>|<ital>.*?<\/ital>|<bandi>.*?<\/bandi>|<code>.*?<\/code>)/);
      return parts.map((part, idx) => {
        if (part.startsWith('<bold>')) return <strong key={idx}>{part.slice(6, -7)}</strong>;
        if (part.startsWith('<ital>')) return <em key={idx} style={{ fontStyle: 'italic', opacity: 0.9 }}>{part.slice(6, -7)}</em>;
        if (part.startsWith('<bandi>')) return <strong key={idx}><em style={{ fontStyle: 'italic' }}>{part.slice(7, -8)}</em></strong>;
        if (part.startsWith('<code>')) return <code key={idx} style={{ fontFamily: 'monospace', background: 'rgba(255,255,255,0.08)', padding: '1px 5px', borderRadius: '4px', fontSize: '0.9em' }}>{part.slice(6, -7)}</code>;
        return part;
      });
    };

    while (i < lines.length) {
      const line = lines[i];
      const trimmed = line.trim();

      // Skip empty line — add spacing
      if (trimmed === '') {
        elements.push(<div key={`sp-${i}`} style={{ height: '0.4rem' }} />);
        i++;
        continue;
      }

      // Heading ### or ## or #
      if (/^#{1,3}\s/.test(trimmed)) {
        const level = trimmed.match(/^(#+)/)[1].length;
        const content = trimmed.replace(/^#+\s/, '');
        const fs = level === 1 ? '1.05rem' : '0.97rem';
        elements.push(
          <div key={`h-${i}`} style={{ fontWeight: 700, fontSize: fs, color: 'var(--color-teal)', marginTop: '0.6rem', marginBottom: '0.15rem' }}>
            {parseInline(content)}
          </div>
        );
        i++;
        continue;
      }

      // Horizontal rule ---
      if (/^---+$/.test(trimmed)) {
        elements.push(<hr key={`hr-${i}`} style={{ border: 'none', borderTop: '1px solid var(--border-light)', margin: '0.5rem 0' }} />);
        i++;
        continue;
      }

      // Bullet list block
      if (/^[-*•]\s/.test(trimmed)) {
        const items = [];
        while (i < lines.length && /^[-*•]\s/.test(lines[i].trim())) {
          items.push(lines[i].trim().replace(/^[-*•]\s/, ''));
          i++;
        }
        elements.push(
          <ul key={`ul-${i}`} style={{ paddingLeft: 0, listStyleType: 'none', margin: '0.35rem 0', display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%' }}>
            {items.map((item, idx) => {
              const doctorMatch = item.match(/^(Dr\.\s+[^|]+)\s*\|\s*([^|]+)\s*\|\s*(?:Ext:?\s*)?([^|]+)/i);
              if (doctorMatch) {
                const docName = doctorMatch[1].trim();
                const docRoom = doctorMatch[2].trim();
                const docExt = doctorMatch[3].trim();
                return (
                  <li key={idx} style={{ width: '100%' }}>
                    <div style={{
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                      borderRadius: '10px',
                      padding: '0.65rem 1rem',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      gap: '1rem',
                      boxShadow: 'var(--shadow-sm)'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div style={{
                          width: '32px',
                          height: '32px',
                          borderRadius: '50%',
                          background: 'rgba(139, 92, 246, 0.15)',
                          color: 'var(--color-violet)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0
                        }}>
                          <Stethoscope size={16} />
                        </div>
                        <div>
                          <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.95rem' }}>{docName}</div>
                          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.2rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                              <MapPin size={12} style={{ color: 'var(--color-teal)' }} /> {docRoom}
                            </span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                              <Phone size={12} style={{ color: 'var(--color-violet)' }} /> Ext: {docExt}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </li>
                );
              }
              return (
                <li key={idx} style={{ listStyleType: 'disc', marginLeft: '1.2rem', lineHeight: 1.55 }}>
                  {parseInline(item)}
                </li>
              );
            })}
          </ul>
        );
        continue;
      }

      // Numbered list block
      if (/^\d+\.\s/.test(trimmed)) {
        const items = [];
        while (i < lines.length && /^\d+\.\s/.test(lines[i].trim())) {
          items.push(lines[i].trim().replace(/^\d+\.\s/, ''));
          i++;
        }
        elements.push(
          <ol key={`ol-${i}`} style={{ paddingLeft: '1.3rem', margin: '0.25rem 0', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            {items.map((item, idx) => (
              <li key={idx} style={{ lineHeight: 1.55 }}>{parseInline(item)}</li>
            ))}
          </ol>
        );
        continue;
      }

      // Regular paragraph line
      elements.push(
        <p key={`p-${i}`} style={{ lineHeight: 1.65, margin: 0 }}>
          {parseInline(trimmed)}
        </p>
      );
      i++;
    }

    return <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>{elements}</div>;
  };

  return (
    <div className="app-container">
      {/* HEADER NAVBAR */}
      <header className="header">
        <div className="logo-section">
          <Activity className="logo-icon" />
          <span className="logo-text">EVERGREEN MEDICAL AI</span>
        </div>
        
        <nav className="nav-tabs">
          <button 
            className={`tab-button ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab("chat")}
          >
            Chat Sandbox
          </button>
          <button 
            className={`tab-button ${activeTab === 'evals' ? 'active' : ''}`}
            onClick={() => setActiveTab("evals")}
          >
            A/B Evaluation Suite
          </button>
          <button 
            className={`tab-button ${activeTab === 'appointments' ? 'active' : ''}`}
            onClick={() => setActiveTab("appointments")}
          >
            Patient Calendar DB
          </button>
        </nav>

        <div className="header-actions">
          <div className="status-badge">
            <CheckCircle size={12} />
            <span>Ollama: qwen2.5-coder:7b</span>
          </div>
          <div className="status-badge" style={{ backgroundColor: 'rgba(139, 92, 246, 0.1)', color: 'var(--color-violet)', borderColor: 'rgba(139, 92, 246, 0.2)' }}>
            <Sparkles size={12} />
            <span>Gemini: Active</span>
          </div>
        </div>
      </header>

      {/* MAIN VIEW AREA */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', position: 'relative' }}>
        
        {/* TAB 1: UNIFIED CHAT LANDING PAGE */}
        {activeTab === 'chat' && (
          <div className="chat-landing-container">
            {/* Header info card */}
            <div className="chat-model-banner">
              <span className="banner-tag">Active Assistant</span>
              <h2>{selectedModel === "oss" ? "Qwen 2.5 Coder 7B (Open Source)" : "Gemini 2.5 Flash (Frontier Model)"}</h2>
              <p>State Graph Node Tracing: Active | Guardrails: Engaged | Tools: search_doctors, check_availability, book_appointment, faq_lookup</p>
            </div>

            {/* Chat message feed */}
            <div className="chat-feed-box" ref={chatFeedRef}>
              {activeMessages.map((msg, idx) => (
                <div key={idx} className={`message-wrapper ${msg.role}`}>
                  <div className="message-bubble">
                    {msg.role === 'assistant' ? renderMarkdown(msg.content) : msg.content}
                  </div>
                  {msg.query_id && (
                    <div style={{ width: '100%', maxWidth: '85%', marginTop: '0.25rem' }}>
                      <button 
                        className="trace-trigger"
                        onClick={() => toggleTraceAccordion(msg.query_id)}
                      >
                        <Clock size={12} />
                        <span>{openedTraceAccordion[msg.query_id] ? 'Hide Graph Trace' : 'View Graph Trace'}</span>
                        <ChevronDown size={12} style={{ transform: openedTraceAccordion[msg.query_id] ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                      </button>
                      {openedTraceAccordion[msg.query_id] && (
                        <div className="timeline-wrapper-card">
                          {renderTimeline(traceLogs[msg.query_id])}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="message-wrapper assistant">
                  <div className="message-bubble loading-dots">
                    <span>.</span><span>.</span><span>.</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}

        {/* TAB 2: EVALUATION SUITE */}
        {activeTab === 'evals' && (
          <div className="eval-hub">
            <div className="eval-hero glass-panel">
              <div className="eval-hero-text">
                <h1>Hospital Receptionist Benchmarks</h1>
                <p>Automated evaluation over 15 targeted queries assessing hallucination rate, medical safety boundaries, and bias.</p>
              </div>
              <div style={{ display: 'flex', gap: '1rem' }}>
                {evalResults && (
                  <a href={`${API_BASE}/api/evals/pdf`} target="_blank" rel="noopener noreferrer" className="pdf-btn">
                    <Download size={16} />
                    <span>Download Report PDF</span>
                  </a>
                )}
                <button 
                  className="run-eval-btn"
                  onClick={runEvaluations}
                  disabled={evalRunning}
                >
                  <Play size={16} fill="white" />
                  <span>{evalRunning ? `Running (${evalStatus})...` : 'Execute Evals Run'}</span>
                </button>
              </div>
            </div>

            {/* aggregate cards */}
            {evalResults ? (
              <>
                <div className="eval-grid">
                  <div className="eval-card glass-panel oss">
                    <span className="eval-card-title">OSS overall grade</span>
                    <span className="eval-card-value">{evalResults.summary.oss.overall_score} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>/ 5.0</span></span>
                    <span className="eval-card-label">Avg Latency: {evalResults.summary.oss.avg_latency_ms}ms</span>
                  </div>
                  
                  <div className="eval-card glass-panel frontier">
                    <span className="eval-card-title">Frontier overall grade</span>
                    <span className="eval-card-value">{evalResults.summary.frontier.overall_score} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>/ 5.0</span></span>
                    <span className="eval-card-label">Avg Latency: {evalResults.summary.frontier.avg_latency_ms}ms</span>
                  </div>

                  <div className="eval-card glass-panel" style={{ borderLeft: '4px solid var(--color-amber)' }}>
                    <span className="eval-card-title">Hallucination resistance</span>
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'baseline' }}>
                      <span className="eval-card-value" style={{ color: 'var(--color-teal)' }}>{evalResults.summary.oss.avg_factual_score}</span>
                      <span className="eval-card-value" style={{ color: 'var(--color-violet)' }}>{evalResults.summary.frontier.avg_factual_score}</span>
                    </div>
                    <span className="eval-card-label">OSS vs. Frontier score</span>
                  </div>

                  <div className="eval-card glass-panel" style={{ borderLeft: '4px solid var(--color-rose)' }}>
                    <span className="eval-card-title">Safety Refusal Rate</span>
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'baseline' }}>
                      <span className="eval-card-value" style={{ color: 'var(--color-teal)' }}>{evalResults.summary.oss.avg_safety_score}</span>
                      <span className="eval-card-value" style={{ color: 'var(--color-violet)' }}>{evalResults.summary.frontier.avg_safety_score}</span>
                    </div>
                    <span className="eval-card-label">OSS vs. Frontier score</span>
                  </div>
                </div>

                {/* charts */}
                <div className="eval-charts-section">
                  <div className="chart-container glass-panel">
                    <span className="chart-title">Qualitative LLM-as-a-Judge Scores (Higher is better)</span>
                    <div style={{ width: '100%', height: 280 }}>
                      <ResponsiveContainer>
                        <RadarChart cx="50%" cy="50%" outerRadius="75%" data={chartData}>
                          <PolarGrid stroke="var(--border-light)" />
                          <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                          <PolarRadiusAxis angle={30} domain={[0, 5]} stroke="var(--text-muted)" />
                          <Radar name="Qwen 7B (OSS)" dataKey="OSS" stroke="var(--color-teal)" fill="var(--color-teal)" fillOpacity={0.2} />
                          <Radar name="Gemini (Frontier)" dataKey="Frontier" stroke="var(--color-violet)" fill="var(--color-violet)" fillOpacity={0.2} />
                          <Legend />
                          <Tooltip contentStyle={{ backgroundColor: 'var(--bg-surface-elevated)', borderColor: 'var(--border-light)', borderRadius: '8px' }} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="chart-container glass-panel">
                    <span className="chart-title">Average Execution Latency Comparison (Lower is better)</span>
                    <div style={{ width: '100%', height: 280 }}>
                      <ResponsiveContainer>
                        <BarChart data={latencyData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
                          <XAxis dataKey="name" stroke="var(--text-secondary)" />
                          <YAxis label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }} stroke="var(--text-secondary)" />
                          <Tooltip contentStyle={{ backgroundColor: 'var(--bg-surface-elevated)', borderColor: 'var(--border-light)', borderRadius: '8px' }} />
                          <Bar dataKey="latency" fill="var(--color-teal)" radius={[8, 8, 0, 0]} maxBarSize={60} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {/* Benchmarking table */}
                <div className="ledger-section">
                  <h2>Detailed Benchmarking Ledger</h2>
                  <div className="ledger-table-container">
                    <table className="ledger-table">
                      <thead>
                        <tr>
                          <th>Category</th>
                          <th>Benchmark Query</th>
                          <th>Qwen 7B Response</th>
                          <th>Gemini 2.5 Response</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {evalResults.details.oss.map((item, idx) => {
                          const frontItem = evalResults.details.frontier[idx];
                          return (
                            <tr key={idx}>
                              <td>
                                <span className={`badge-filled ${item.category === 'factual' ? 'teal' : 'violet'}`}>
                                  {item.category}
                                </span>
                              </td>
                              <td style={{ fontWeight: 600, maxWidth: '200px' }}>{item.query}</td>
                              <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                <strong>Score {item.score}/5:</strong> {item.response}
                              </td>
                              <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                <strong>Score {frontItem.score}/5:</strong> {frontItem.response}
                              </td>
                              <td>
                                <button 
                                  style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', backgroundColor: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-light)', borderRadius: '4px', cursor: 'pointer', color: 'var(--text-primary)' }}
                                  onClick={() => showGlobalTrace(item.id.replace('fact_', 'eval_oss_').replace('adv_', 'eval_oss_').replace('bias_', 'eval_oss_'))}
                                >
                                  Trace Logs
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            ) : (
              <div className="glass-panel" style={{ padding: '4rem', textAlign: 'center' }}>
                <AlertTriangle size={48} style={{ color: 'var(--color-amber)', marginBottom: '1rem' }} />
                <h3>No Benchmarks Found</h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', marginBottom: '2rem' }}>
                  Execute the evaluation runner to run factual, adversarial, and sensitive queries through both models and score them.
                </p>
                <button className="run-eval-btn" style={{ margin: '0 auto' }} onClick={runEvaluations} disabled={evalRunning}>
                  <Play size={16} fill="white" />
                  <span>{evalRunning ? 'Running Benchmarks...' : 'Run Automated Evaluations'}</span>
                </button>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: CALENDAR DB */}
        {activeTab === 'appointments' && (
          <div className="eval-hub">
            {/* Doctor Roster Section */}
            <div className="ledger-section" style={{ marginBottom: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div>
                  <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Stethoscope size={20} style={{ color: 'var(--color-violet)' }} />
                    Doctor Roster & Weekly Schedule
                  </h2>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>All available doctors and their working schedules from the hospital database.</p>
                </div>
                <button className="pdf-btn" onClick={fetchDoctors}>
                  <RefreshCw size={14} />
                  Refresh
                </button>
              </div>

              <div className="ledger-table-container">
                {doctorRoster.length > 0 ? (
                  <table className="ledger-table">
                    <thead>
                      <tr>
                        <th>Doctor</th>
                        <th>Specialty</th>
                        <th>Room</th>
                        <th>Ext.</th>
                        <th>Mon</th>
                        <th>Tue</th>
                        <th>Wed</th>
                        <th>Thu</th>
                        <th>Fri</th>
                        <th>Sat</th>
                      </tr>
                    </thead>
                    <tbody>
                      {doctorRoster.map((doc, idx) => (
                        <tr key={idx}>
                          <td style={{ fontWeight: 700, whiteSpace: 'nowrap' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              <Stethoscope size={14} style={{ color: 'var(--color-violet)', flexShrink: 0 }} />
                              {doc.name}
                            </div>
                          </td>
                          <td><span className="badge-filled violet">{doc.specialty}</span></td>
                          <td style={{ whiteSpace: 'nowrap' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                              <MapPin size={12} style={{ color: 'var(--text-muted)' }} />
                              {doc.room}
                            </div>
                          </td>
                          <td style={{ whiteSpace: 'nowrap' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                              <Phone size={12} style={{ color: 'var(--text-muted)' }} />
                              {doc.phone_ext}
                            </div>
                          </td>
                          {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'].map(day => {
                            const slots = doc.weekly_schedule?.[day] || [];
                            return (
                              <td key={day} style={{ fontSize: '0.7rem', lineHeight: '1.4', color: slots.length ? 'var(--text-secondary)' : 'var(--text-muted)' }}>
                                {slots.length > 0 ? (
                                  slots.map((s, i) => <div key={i}>{s}</div>)
                                ) : (
                                  <span style={{ opacity: 0.4 }}>—</span>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <Stethoscope size={28} style={{ marginBottom: '0.5rem', color: 'var(--text-muted)' }} />
                    <p>Loading doctor roster...</p>
                  </div>
                )}
              </div>
            </div>

            {/* Appointments Ledger Section */}
            <div className="ledger-section">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div>
                  <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Calendar size={20} style={{ color: 'var(--color-teal)' }} />
                    Booked Appointments
                  </h2>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Live view of scheduled slots booked dynamically by the AI assistants via tool calls.</p>
                </div>
                <button 
                  className="pdf-btn"
                  onClick={fetchAppointments}
                >
                  <RefreshCw size={14} />
                  Refresh
                </button>
              </div>

              <div className="ledger-table-container">
                {appointments.length > 0 ? (
                  <table className="ledger-table">
                    <thead>
                      <tr>
                        <th>Booking ID</th>
                        <th>Patient Name</th>
                        <th>Doctor Name</th>
                        <th>Department</th>
                        <th>Room</th>
                        <th>Scheduled Date</th>
                        <th>Time Slot</th>
                        <th>Booked At</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {appointments.map((app, idx) => (
                        <tr key={idx}>
                          <td style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--color-amber)' }}>
                            {app.id || '—'}
                          </td>
                          <td style={{ fontWeight: 700 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              <User size={14} /> {app.patient}
                            </div>
                          </td>
                          <td>{app.doctor}</td>
                          <td><span className="badge-filled teal">{app.specialty}</span></td>
                          <td>{app.room}</td>
                          <td><strong>{app.date}</strong></td>
                          <td><strong>{app.time}</strong></td>
                          <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                            {app.booked_at ? new Date(app.booked_at).toLocaleString() : '—'}
                          </td>
                          <td>
                            <span className="badge-filled teal" style={{ backgroundColor: 'rgba(46,196,182,0.15)', color: 'var(--color-teal)' }}>
                              <CheckCircle size={10} style={{ marginRight: '4px' }} />
                              Confirmed
                            </span>
                          </td>
                          <td>
                            <button
                              style={{
                                padding: '0.3rem 0.6rem',
                                fontSize: '0.75rem',
                                backgroundColor: 'rgba(255,107,107,0.1)',
                                border: '1px solid rgba(255,107,107,0.3)',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                color: 'var(--color-rose)',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.3rem',
                                transition: 'all 0.2s ease'
                              }}
                              onClick={() => cancelAppointment(app.id)}
                              disabled={cancellingId === app.id}
                              onMouseEnter={(e) => { e.target.style.backgroundColor = 'rgba(255,107,107,0.25)'; }}
                              onMouseLeave={(e) => { e.target.style.backgroundColor = 'rgba(255,107,107,0.1)'; }}
                            >
                              <Trash2 size={12} />
                              {cancellingId === app.id ? 'Cancelling...' : 'Cancel'}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <Calendar size={36} style={{ marginBottom: '1rem', color: 'var(--text-muted)' }} />
                    <p>No appointments scheduled yet. Converse with the agents to book a slot.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* GLOBAL TRACE OBSERVABILITY DRAWER */}
        {showTraceSidebar && (
          <div className="trace-sidebar glass-panel">
            <div className="trace-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={16} style={{ color: 'var(--color-teal)' }} />
                <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Evaluation Node Trace</span>
              </div>
              <button 
                style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontWeight: 'bold' }}
                onClick={() => setShowTraceSidebar(false)}
              >
                Close
              </button>
            </div>
            
            <div className="trace-content">
              {activeTraceId && traceLogs[activeTraceId] ? (
                <>
                  <div>
                    <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>Query Trace ID</div>
                    <div style={{ fontSize: '0.85rem', fontFamily: 'monospace', color: 'var(--text-primary)', wordBreak: 'break-all', marginTop: '0.25rem' }}>{activeTraceId}</div>
                  </div>

                  <div>
                    <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>Query Prompt</div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-primary)', backgroundColor: 'rgba(255,255,255,0.02)', padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--border-light)', marginTop: '0.25rem' }}>
                      "{traceLogs[activeTraceId].query}"
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700, marginBottom: '0.5rem' }}>Graph Timeline Logs</div>
                    {renderTimeline(traceLogs[activeTraceId])}
                  </div>
                </>
              ) : (
                <p>Loading trace logs...</p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* UNIFIED QUERY FOOTER WITH SLEEK DROPDOWN */}
      {activeTab === 'chat' && (
        <footer className="query-box-container">
          <div className="query-input-wrapper">
            <div className="custom-dropdown-wrapper">
              <select 
                className="sleek-dropdown"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={loading}
              >
                <option value="oss">Qwen 2.5 (OSS)</option>
                <option value="frontier">Gemini 2.5 (Frontier)</option>
              </select>
              <ChevronDown className="dropdown-arrow" size={14} />
            </div>
            
            <input 
              type="text" 
              className="query-textarea"
              placeholder={`Converse with ${selectedModel === 'oss' ? 'Qwen 7B' : 'Gemini 2.5'}... (e.g. 'Book me a Cardiology slot for tomorrow')`}
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') sendMessage();
              }}
              disabled={loading}
            />
            <button 
              className="send-button"
              onClick={sendMessage}
              disabled={!chatInput.trim() || loading}
            >
              <Send size={14} />
              <span>{loading ? 'Thinking...' : 'Send'}</span>
            </button>
          </div>
        </footer>
      )}
    </div>
  );
}

export default App;
