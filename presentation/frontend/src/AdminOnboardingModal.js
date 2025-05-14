import React, { useState, useEffect } from "react";
import "./AdminPanel.css";

function AdminOnboardingModal() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (!localStorage.getItem("admin_onboarding_seen")) {
      setShow(true);
    }
  }, []);

  const handleClose = () => {
    setShow(false);
    localStorage.setItem("admin_onboarding_seen", "1");
  };

  if (!show) return null;
  return (
    <div className="onboarding-modal" role="dialog" aria-modal="true" tabIndex={-1}>
      <div className="modal-content">
        <h2>Welcome to the GoldenSignalsAI Admin Panel!</h2>
        <ul>
          <li>Monitor system health, agents, and queue in real time.</li>
          <li>Manage users and roles securely.</li>
          <li>All sensitive actions are audit-logged for compliance.</li>
          <li>Need help? See the documentation or contact support.</li>
        </ul>
        <button onClick={handleClose} aria-label="Close onboarding modal">Got it!</button>
      </div>
    </div>
  );
}
export default AdminOnboardingModal;
