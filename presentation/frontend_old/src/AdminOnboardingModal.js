// AdminOnboardingModal.js
// Purpose: Displays a modal dialog to onboard new admin users in GoldenSignalsAI. Provides information about the admin panel's features and functionality. Tracks onboarding completion in local storage to avoid repeat displays for the same user.

import React, { useState, useEffect } from "react";
import "./AdminPanel.css";

function AdminOnboardingModal() {
  // State to track whether onboarding modal is visible
  const [show, setShow] = useState(() => {
    // Check local storage to see if onboarding is already complete
    return !localStorage.getItem("admin_onboarding_seen");
  });

  // Handle closing of onboarding modal and mark as complete in local storage
  const handleClose = () => {
    setShow(false);
    localStorage.setItem("admin_onboarding_seen", "1");
  };

  // If onboarding is complete, render nothing
  if (!show) return null;

  // Render onboarding modal with admin panel instructions
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
