import React from 'react';
import { FaTachometerAlt, FaBell, FaListUl, FaHistory, FaUser } from 'react-icons/fa';
import './SidebarNav.css';

const navItems = [
  { icon: <FaTachometerAlt />, label: 'Dashboard', to: '/' },
  { icon: <FaBell />, label: 'Signals', to: '/signals' },
  { icon: <FaListUl />, label: 'Watchlist', to: '/watchlist' },
  { icon: <FaHistory />, label: 'History', to: '/history' },
  { icon: <FaUser />, label: 'Profile', to: '/profile' },
];

export default function SidebarNav({ active, onNavigate }) {
  return (
    <nav className="sidebar-nav">
      <ul>
        {navItems.map((item, idx) => (
          <li key={item.label} className={active === item.label ? 'active' : ''} onClick={() => onNavigate(item.to)}>
            {item.icon}
            <span className="nav-label">{item.label}</span>
          </li>
        ))}
      </ul>
    </nav>
  );
}
