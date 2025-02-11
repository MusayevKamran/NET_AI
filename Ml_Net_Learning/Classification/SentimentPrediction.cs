using Microsoft.ML.Data;

namespace Ml_Net_Learning.Classification;

public class SentimentPrediction
{

    [ColumnName("Score")]
    public float SentimentScore { get; set; }

    public bool IsPositiveSentiment => SentimentScore < 0.5f;
}