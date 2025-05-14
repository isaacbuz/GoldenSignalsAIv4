// Firebase config for GoldenSignalsAI Admin Panel
// Replace the below config with your actual Firebase project credentials
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, GithubAuthProvider, onIdTokenChanged, signOut } from "firebase/auth";

const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID,
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();
const githubProvider = new GithubAuthProvider();

// Helper: Detect token expiry and auto-logout
export function setupTokenExpiryListener(onExpire) {
  onIdTokenChanged(auth, async (user) => {
    if (user) {
      const token = await user.getIdTokenResult();
      const now = Date.now() / 1000;
      if (token.expirationTime && now > Date.parse(token.expirationTime) / 1000) {
        signOut(auth);
        if (onExpire) onExpire();
      }
    } else {
      if (onExpire) onExpire();
    }
  });
}

export { auth, googleProvider, githubProvider };
