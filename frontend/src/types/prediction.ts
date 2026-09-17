export interface TopPrediction {
  ec_class: string;
  probability: number;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  top_predictions: TopPrediction[];
  sequence_length: number;
  disclaimer: string;
}

export interface PresetSample {
  name: string;
  expectedClass: string;
  sequence: string;
}