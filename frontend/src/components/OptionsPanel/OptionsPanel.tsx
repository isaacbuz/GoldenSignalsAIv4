import React from 'react';

export interface OptionsPanelProps {
  options: Array<{ price: number; optionDetails: string }>;
}

export const OptionsPanel: React.FC<OptionsPanelProps> = ({ options }) => {
  return (
    <div className="options-panel">
      <h3>Recommended Options</h3>
      <ul>
        {options.map((opt, idx) => (
          <li key={idx}>{opt.optionDetails} at ${opt.price}</li>
        ))}
      </ul>
    </div>
  );
};
