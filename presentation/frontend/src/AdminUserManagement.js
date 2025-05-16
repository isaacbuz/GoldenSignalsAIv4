// AdminUserManagement.js
// Purpose: Provides user management functionality for GoldenSignalsAI administrators. Allows listing, inviting, enabling, disabling, and bulk managing users via backend API calls. Handles user feedback, loading state, and error reporting for robust admin operations.

import React, { useEffect, useState } from "react";
import "./AdminPanel.css";

function AdminUserManagement() {
  // State for list of users
  const [users, setUsers] = useState([]);
  // State for selected users (for bulk actions)
  const [selected, setSelected] = useState([]);
  // State for loading indicator
  const [loading, setLoading] = useState(false);
  // State for error messages
  const [error, setError] = useState("");
  // State for feedback messages
  const [msg, setMsg] = useState("");
  // State for new user email input
  const [inviteEmail, setInviteEmail] = useState("");
  // State for inviting a new user
  const [inviting, setInviting] = useState(false);
  // State for resetting a user's password
  const [resettingUid, setResettingUid] = useState("");
  // State for deleting a user
  const [deletingUid, setDeletingUid] = useState("");
  // State for bulk actions
  const [selectAll, setSelectAll] = useState(false);
  const [bulkAction, setBulkAction] = useState("");
  const [bulkLoading, setBulkLoading] = useState(false);

  // Fetch user list from backend on mount
  useEffect(() => {
    fetch("/api/admin/users/")
      .then((res) => res.json())
      .then((data) => {
        setUsers(data);
        setLoading(false);
      });
  }, []);

  // Handle setting a user's role
  const handleSetRole = async (uid, role) => {
    setMsg("");
    const res = await fetch(`/api/admin/users/${uid}/set_role?role=${role}`, { method: "POST" });
    const data = await res.json();
    setMsg(data.message);
  };

  // Handle disabling a user
  const handleDisable = async (uid) => {
    setMsg("");
    const res = await fetch(`/api/admin/users/${uid}/disable`, { method: "POST" });
    const data = await res.json();
    setMsg(data.message);
  };

  // Handle enabling a user
  const handleEnable = async (uid) => {
    setMsg("");
    const res = await fetch(`/api/admin/users/${uid}/enable`, { method: "POST" });
    const data = await res.json();
    setMsg(data.message);
  };

  // Handle inviting a new user by email
  const handleInvite = async () => {
    if (!inviteEmail) return;
    setInviting(true);
    setMsg("");
    const res = await fetch(`/api/admin/users/invite`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: inviteEmail })
    });
    const data = await res.json();
    setMsg(data.message);
    setInviteEmail("");
    setInviting(false);
  };

  // Handle resetting a user's password
  const handleResetPassword = async (uid) => {
    setResettingUid(uid);
    setMsg("");
    const res = await fetch(`/api/admin/users/${uid}/reset_password`, { method: "POST" });
    const data = await res.json();
    setMsg(data.message);
    setResettingUid("");
  };

  // Handle deleting a user
  const handleDelete = async (uid, email) => {
    if (!window.confirm(`Are you sure you want to delete user ${email}? This cannot be undone.`)) return;
    setDeletingUid(uid);
    setMsg("");
    const res = await fetch(`/api/admin/users/${uid}/delete`, { method: "POST" });
    const data = await res.json();
    setMsg(data.message);
    setDeletingUid("");
  };

  // Handle user selection for bulk actions
  const handleSelect = (uid) => {
    setSelected(selected.includes(uid) ? selected.filter(id => id !== uid) : [...selected, uid]);
  };

  // Handle selecting all users for bulk actions
  const handleSelectAll = () => {
    if (selectAll) {
      setSelected([]);
      setSelectAll(false);
    } else {
      setSelected(users.map(u => u.uid));
      setSelectAll(true);
    }
  };

  // Handle bulk actions
  const handleBulkAction = async () => {
    if (!bulkAction || selected.length === 0) return;
    setBulkLoading(true);
    setMsg("");
    let endpoint = "";
    if (bulkAction === "enable") endpoint = "/api/admin/users/bulk_enable";
    if (bulkAction === "disable") endpoint = "/api/admin/users/bulk_disable";
    if (bulkAction === "delete") endpoint = "/api/admin/users/bulk_delete";
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(selected)
    });
    const data = await res.json();
    setMsg(`${bulkAction.charAt(0).toUpperCase() + bulkAction.slice(1)}: ${data.results.filter(r => r.success).length} succeeded, ${data.results.filter(r => !r.success).length} failed.`);
    setBulkLoading(false);
    setSelected([]);
    setSelectAll(false);
  };

  // Render user management UI: invite, feedback, user table, and bulk actions
  if (loading) return <p>Loading users...</p>;

  return (
    <div className="admin-users">
      <h4>User & Role Management</h4>
      <div style={{ marginBottom: "1rem" }}>
        <input
          type="email"
          placeholder="Invite user by email"
          value={inviteEmail}
          onChange={e => setInviteEmail(e.target.value)}
          style={{ padding: "0.5rem", borderRadius: 6, border: "1px solid #333", marginRight: 8 }}
        />
        <button
          onClick={handleInvite}
          disabled={inviting || !inviteEmail}
          style={{ background: "#6be6c1", color: "#222", border: "none", borderRadius: 6, padding: "0.5rem 1rem", cursor: "pointer" }}
        >
          {inviting ? "Inviting..." : "Invite User"}
        </button>
      </div>
      <div style={{ marginBottom: "1rem" }}>
        <input type="checkbox" checked={selectAll} onChange={handleSelectAll} /> Select All
        <select value={bulkAction} onChange={e => setBulkAction(e.target.value)} style={{ marginLeft: 8, marginRight: 8 }}>
          <option value="">Bulk Action</option>
          <option value="enable">Enable</option>
          <option value="disable">Disable</option>
          <option value="delete">Delete</option>
        </select>
        <button onClick={handleBulkAction} disabled={bulkLoading || !bulkAction || selected.length === 0} style={{ background: "#f8b400", color: "#222", border: "none", borderRadius: 6, padding: "0.4rem 1rem", cursor: "pointer" }}>
          {bulkLoading ? "Processing..." : "Apply to Selected"}
        </button>
      </div>
      {msg && <div style={{ margin: "1rem 0", color: "#6be6c1", fontWeight: "bold" }}>{msg}</div>}
      <table className="admin-users-table">
        <thead>
          <tr>
            <th><input type="checkbox" checked={selectAll} onChange={handleSelectAll} /></th>
            <th>Email</th>
            <th>Display Name</th>
            <th>Role</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {users.map((u) => (
            <tr key={u.uid}>
              <td><input type="checkbox" checked={selected.includes(u.uid)} onChange={() => handleSelect(u.uid)} /></td>
              <td>{u.email}</td>
              <td>{u.displayName}</td>
              <td>{(u.customClaims && u.customClaims.role) || "user"}</td>
              <td style={{ color: u.disabled ? "#ff5252" : "#6be6c1" }}>{u.disabled ? "Disabled" : "Active"}</td>
              <td>
                <select
                  defaultValue={(u.customClaims && u.customClaims.role) || "user"}
                  onChange={(e) => handleSetRole(u.uid, e.target.value)}
                  style={{ marginRight: 8 }}
                >
                  <option value="user">user</option>
                  <option value="admin">admin</option>
                  <option value="operator">operator</option>
                </select>
                {u.disabled ? (
                  <button onClick={() => handleEnable(u.uid)} style={{ background: "#6be6c1", color: "#222", border: "none", borderRadius: 6, padding: "0.3rem 0.7rem", cursor: "pointer", marginRight: 8 }}>Enable</button>
                ) : (
                  <button onClick={() => handleDisable(u.uid)} style={{ background: "#ff5252", color: "#fff", border: "none", borderRadius: 6, padding: "0.3rem 0.7rem", cursor: "pointer", marginRight: 8 }}>Disable</button>
                )}
                <button
                  onClick={() => handleResetPassword(u.uid)}
                  disabled={resettingUid === u.uid}
                  style={{ background: "#f8b400", color: "#222", border: "none", borderRadius: 6, padding: "0.3rem 0.7rem", cursor: "pointer", marginRight: 8 }}
                >
                  {resettingUid === u.uid ? "Resetting..." : "Reset Password"}
                </button>
                <button
                  onClick={() => handleDelete(u.uid, u.email)}
                  disabled={deletingUid === u.uid}
                  style={{ background: "#292929", color: "#fff", border: "none", borderRadius: 6, padding: "0.3rem 0.7rem", cursor: "pointer" }}
                >
                  {deletingUid === u.uid ? "Deleting..." : "Delete"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default AdminUserManagement;
