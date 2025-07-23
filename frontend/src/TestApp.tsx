import React from 'react';

const TestApp: React.FC = () => {
  console.log('TestApp rendering');

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      backgroundColor: '#000',
      color: '#fff',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'column'
    }}>
      <h1>GoldenSignalsAI Test</h1>
      <p>If you see this, React is working!</p>
      <div style={{ marginTop: '20px', padding: '20px', backgroundColor: '#111', borderRadius: '8px' }}>
        <p>Debug Info:</p>
        <p>Time: {new Date().toLocaleTimeString()}</p>
        <p>URL: {window.location.href}</p>
      </div>
    </div>
  );
};

export default TestApp;
