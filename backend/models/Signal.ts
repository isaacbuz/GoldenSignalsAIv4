import mongoose from 'mongoose';

const SignalSchema = new mongoose.Schema({
  symbol: String,
  type: String,
  confidence: Number,
  entry: Number,
  createdAt: { type: Date, default: Date.now },
});

export default mongoose.models.Signal || mongoose.model('Signal', SignalSchema);
