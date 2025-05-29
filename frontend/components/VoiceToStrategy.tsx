import { useState, useEffect } from 'react';

export default function VoiceToStrategy({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [listening, setListening] = useState(false);
  const [text, setText] = useState('');

  useEffect(() => {
    // @ts-ignore
    if (!('webkitSpeechRecognition' in window)) return;
    // @ts-ignore
    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (e: any) => {
      const transcript = e.results[0][0].transcript;
      setText(transcript);
      onSubmit(transcript);
    };

    if (listening) recognition.start();
    else recognition.stop();

    return () => recognition.stop();
  }, [listening, onSubmit]);

  return (
    <div className="bg-bgPanel p-4 rounded-lg text-white space-y-2 font-sans" aria-label="Voice to strategy input">
      {/*
        VoiceToStrategy enables users to dictate a trading strategy using voice input.
      */}
      <button
        onClick={() => setListening(!listening)}
        className={`w-full py-2 rounded font-bold font-sans transition ${
          listening ? 'bg-accentPink' : 'bg-accentGreen text-black hover:bg-accentGreen/80'
        }`}
        aria-label={listening ? 'Stop listening' : 'Start voice input'}
      >
        {listening ? ' Stop Listening' : ' Start Voice Input'}
      </button>
      {text && <p className="italic text-gray-400" aria-label="Heard text">Heard: "{text}"</p>}
    </div>
  );
}
