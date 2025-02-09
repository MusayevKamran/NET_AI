using Microsoft.ML.Data;

namespace Ml_Net_Learning.Housing;

public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}