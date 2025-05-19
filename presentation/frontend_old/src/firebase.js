// Firebase config for GoldenSignalsAI Admin Panel
// Replace the below config with your actual Firebase project credentials
// firebase.js
// Purpose: Initializes and exports Firebase app and authentication providers for GoldenSignalsAI frontend. Centralizes Firebase configuration and ensures secure, maintainable integration with Firebase services.

import { setPersistence, browserLocalPersistence } from "firebase/auth";
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, GithubAuthProvider, onIdTokenChanged, signOut } from "firebase/auth";

// Firebase configuration object for the GoldenSignalsAI project
const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID,
  measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID,
};

// Initialize Firebase app instance
const app = initializeApp(firebaseConfig);
// Initialize Firebase authentication instance
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();
const githubProvider = new GithubAuthProvider();

// Helper: Detect token expiry and auto-logout

// Set session persistence to survive page reloads
setPersistence(auth, browserLocalPersistence);

// Helper: Detect token expiry and auto-logout
export function setupTokenExpiryListener(onExpire) {
  onIdTokenChanged(auth, async (user) => {
    if (user) {
      const token = await user.getIdTokenResult();
      const now = Date.now() / 1000;
      // Only sign out if token is truly expired
      if (token.expirationTime && now > Date.parse(token.expirationTime) / 1000) {
        signOut(auth);
        if (onExpire) onExpire();
      }
    } else {
      // If the user is not authenticated, call the onExpire callback.
      if (onExpire) onExpire();
    }
  });
}

// Export initialized Firebase app and authentication providers for use in the frontend
export { app, auth, googleProvider, githubProvider };
