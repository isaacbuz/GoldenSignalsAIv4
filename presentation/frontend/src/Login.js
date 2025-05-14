import React, { useState } from "react";
import { auth, googleProvider, githubProvider } from "./firebase";
import { signInWithEmailAndPassword, signInWithPopup } from "firebase/auth";
import "./Login.css";

function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleEmailLogin = (e) => {
    e.preventDefault();
    signInWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        onLogin(userCredential.user);
      })
      .catch((err) => setError(err.message));
  };

  const handleGoogleLogin = () => {
    signInWithPopup(auth, googleProvider)
      .then((result) => {
        onLogin(result.user);
      })
      .catch((err) => setError(err.message));
  };

  const handleGithubLogin = () => {
    signInWithPopup(auth, githubProvider)
      .then((result) => {
        onLogin(result.user);
      })
      .catch((err) => setError(err.message));
  };

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
