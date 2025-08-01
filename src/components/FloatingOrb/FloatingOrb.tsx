import { motion } from 'framer-motion';
import { useEffect } from 'react';

interface FloatingOrbProps {
    text: string;
}
const FloatingOrb = ({ text }: FloatingOrbProps) => {
    const speak = (text: string) => {
        // Check if browser supports speech synthesis
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            speechSynthesis.speak(utterance);
        }
    };

    useEffect(() => {
        speak(text);
    }, [text]);

    return (
        <motion.div
            onClick={() => speak('Buy signal detected with 85% confidence')}
            className="absolute top-4 right-4 w-16 h-16 rounded-full bg-golden-primary flex items-center justify-center text-dark-bg"
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
        >
            AI Orb
        </motion.div>
    );
};

export default FloatingOrb;
