import { useNavigate } from "react-router-dom";
import "../styles/login.css";
import logo from "../assets/robot.png";

import '../styles/home.css';
import tamuLogo from '../assets/logos/tamu.png';

export default function LoginChooserPage() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      {/* TOP: Texas A&M Logo */}
      <header className="home-header">
        <img src={tamuLogo} alt="Texas A&M University" className="tamu-logo" />
      </header>

      <div className="aura-login login-page home-content">
        <div className="login-card">
          <div className="login-brand">
            <img src={logo} alt="AURA" className="login-logo" />
            <div className="login-brand-text">
              <h1 className="login-title">AURA</h1>
              <p className="login-subtitle">Choose login type</p>
            </div>
          </div>

          <div className="login-form">
            <button className="login-btn" type="button" onClick={() => navigate("/login/student")}>
              Student Login
            </button>

            <button className="login-btn" type="button" onClick={() => navigate("/login/ta")}>
              TA Login
            </button>

            <button
              className="login-btn login-btn-secondary"
              type="button"
              onClick={() => navigate("/login/admin")}
            >
              Admin Login
            </button>

            <div className="login-footnote">
              Students/TAs: email + 2FA • Admin: email + password + 2FA
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}