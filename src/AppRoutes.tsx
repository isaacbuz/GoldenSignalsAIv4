import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import App from './App';
// Placeholder components (generate full pages later)
const StockView = () => React.createElement('div', null, `Stock Analysis for ${window.location.pathname.split('/')[2]}`);
const OptionsView = () => <div>Options Chain for {window.location.pathname.split('/')[2]}</div>;
const PredictionsView = () => <div>Prediction Comparisons</div>;

const AppRoutes = () => (
    <Router>
        <Routes>
            <Route path="/" element={<App />} />
            <Route path="/dashboard" element={<App />} />
            <Route path="/stocks/:symbol" element={<StockView />} />
            <Route path="/options/:symbol" element={<OptionsView />} />
            <Route path="/predictions" element={<PredictionsView />} />
        </Routes>
    </Router>
);

export default AppRoutes;
