import React, { createContext, useContext, useState, useEffect } from "react";

type Role = "user" | "admin";

interface AuthContextType {
  user: string | null;
  role: Role | null;
  login: (username: string, role: Role) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  role: null,
  login: () => {},
  logout: () => {},
});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<string | null>(null);
  const [role, setRole] = useState<Role | null>(null);

  useEffect(() => {
    const savedUser = localStorage.getItem("gsai-user");
    const savedRole = localStorage.getItem("gsai-role") as Role | null;
    if (savedUser && savedRole) {
      setUser(savedUser);
      setRole(savedRole);
    }
  }, []);

  const login = (username: string, userRole: Role) => {
    setUser(username);
    setRole(userRole);
    localStorage.setItem("gsai-user", username);
    localStorage.setItem("gsai-role", userRole);
  };

  const logout = () => {
    setUser(null);
    setRole(null);
    localStorage.removeItem("gsai-user");
    localStorage.removeItem("gsai-role");
  };

  return (
    <AuthContext.Provider value={{ user, role, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
