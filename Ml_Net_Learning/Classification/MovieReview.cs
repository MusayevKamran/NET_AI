using Microsoft.ML.Data;

namespace Ml_Net_Learning.Classification;

public class MovieReview
{
    [LoadColumn(0)]
    public string Text { get; set; }
    [LoadColumn(1)]
    [ColumnName("Label")]
    public bool Sentiment { get; set; }

}