import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../services/authService";
import "../styles/login.css";
import logo from "../assets/robot.png";
import AdminOtpModal from "../components/AdminOtpModal";

function isTamuEmail(email: string) {
  return email.trim().toLowerCase().endsWith("@tamu.edu");
}

export default function LoginTAPage() {
  const { taStartLogin, taVerifyOtp } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [showOtp, setShowOtp] = useState(false);
  const [otpEmail, setOtpEmail] = useState("");
  const [otpError, setOtpError] = useState<string | null>(null);

  const start = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setNotice(null);
    setLoading(true);

    try {
      const normalizedEmail = email.trim().toLowerCase();
      if (!isTamuEmail(normalizedEmail)) {
        throw new Error("Please use your @tamu.edu email.");
      }

      const hint = await taStartLogin(normalizedEmail);

      setOtpEmail(normalizedEmail);
      setOtpError(null);
      setNotice(hint.notice || null);
      setShowOtp(true);
    } catch (err: any) {
      setError(err?.message || "TA login failed.");
    } finally {
      setLoading(false);
    }
  };

  const verify = async (otp: string) => {
    setOtpError(null);

    try {
      const hint = await taVerifyOtp(otpEmail, otp);
      setNotice(hint.notice || notice);
      setShowOtp(false);
      navigate("/database", { replace: true });
    } catch (err: any) {
      setOtpError(err?.message || "Invalid OTP");
    }
  };

  return (
    <div className="aura-login login-page">
      <div className="login-card">
        <div className="login-brand">
          <img src={logo} alt="AURA" className="login-logo" />
          <div className="login-brand-text">
            <h1 className="login-title">AURA</h1>
            <p className="login-subtitle">TA Login</p>
          </div>
        </div>

        <form onSubmit={start} className="login-form">
          <label className="login-label">TAMU Email</label>
          <input
            className="login-input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="netid@tamu.edu"
            autoComplete="email"
          />

          {error && <div className="login-error">{error}</div>}

          {notice && (
            <div
              style={{
                marginTop: 10,
                padding: "10px 12px",
                borderRadius: 10,
                background: "rgba(255, 193, 7, 0.14)",
                border: "1px solid rgba(255, 193, 7, 0.38)",
                color: "#6b4f00",
                fontSize: "0.95rem",
                lineHeight: 1.4,
              }}
            >
              {notice}
            </div>
          )}

          <button className="login-btn" type="submit" disabled={loading}>
            {loading ? "Sending code..." : "Send Code"}
          </button>

          <div className="login-footnote">TAs: admin approval + email + 2FA</div>
        </form>
      </div>

      {showOtp && (
        <AdminOtpModal
          title="TA verification"
          email={otpEmail}
          error={otpError}
          onCancel={() => setShowOtp(false)}
          onVerify={verify}
        />
      )}
    </div>
  );
}
