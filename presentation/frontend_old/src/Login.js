// Login.js
// Purpose: Provides the login form and authentication logic for GoldenSignalsAI users and admins. Handles user input, form submission, and feedback for failed login attempts. Integrates with backend or Firebase for authentication.

// Import necessary dependencies
import React, { useState } from "react";
import { auth, googleProvider, githubProvider } from "./firebase";
import { signInWithEmailAndPassword, signInWithPopup } from "firebase/auth";
import "./Login.css";

// Define the Login component
function Login({ onLogin }) {
  // State for email and password input
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  // State for error messages
  const [error, setError] = useState("");

  // Handle login form submission
  const handleEmailLogin = (e) => {
    // Prevent default form submission behavior
    e.preventDefault();
    // Call Firebase's signInWithEmailAndPassword function to authenticate user
    signInWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        // Call the onLogin callback function with the authenticated user
        onLogin(userCredential.user);
      })
      .catch((err) => {
        // Set error message if authentication fails
        setError(err.message);
      });
  };

  // Handle Google login button click
  const handleGoogleLogin = () => {
    // Call Firebase's signInWithPopup function to authenticate user with Google
    signInWithPopup(auth, googleProvider)
      .then((result) => {
        // Call the onLogin callback function with the authenticated user
        onLogin(result.user);
      })
      .catch((err) => {
        // Set error message if authentication fails
        setError(err.message);
      });
  };

  // Handle GitHub login button click
  const handleGithubLogin = () => {
    // Call Firebase's signInWithPopup function to authenticate user with GitHub
    signInWithPopup(auth, githubProvider)
      .then((result) => {
        // Call the onLogin callback function with the authenticated user
        onLogin(result.user);
      })
      .catch((err) => {
        // Set error message if authentication fails
        setError(err.message);
      });
  };

  // Render login form UI
  return (
    <div className="login-panel">
      <h2>Admin Login</h2>
      <form onSubmit={handleEmailLogin}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit">Login</button>
      </form>
      <div className="oauth-buttons">
        <button onClick={handleGoogleLogin}>Sign in with Google</button>
        <button onClick={handleGithubLogin}>Sign in with GitHub</button>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default Login;
