using Microsoft.ML.Data;

namespace Ml_Net_Learning.Classification;

public class TextData
{
    [LoadColumn(0)]
    public string Text { get; set; }
}