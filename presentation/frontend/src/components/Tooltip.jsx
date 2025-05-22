import React, { useState } from 'react';
import './Tooltip.css';

export default function Tooltip({ text, children }) {
  const [show, setShow] = useState(false);
  return (
    <span
      className="tooltip-container"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      tabIndex={0}
      onFocus={() => setShow(true)}
      onBlur={() => setShow(false)}
    >
      {children}
      {show && <span className="tooltip-text">{text}</span>}
    </span>
  );
}
