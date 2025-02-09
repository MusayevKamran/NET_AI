using Microsoft.ML.Data;

namespace Ml_Net_Learning.DataPoint;

public class Prediction
{
    [ColumnName("Score")]
    public float Score { get; set; }
    [ColumnName("Probability")]
    public float Probability { get; set; }
    
    public static void EvaluateMetrics(string modelName, BinaryClassificationMetrics metrics)
    {
        Console.WriteLine($"************************************************************");
        Console.WriteLine($"*       Metrics for {modelName} binary classification model      ");
        Console.WriteLine($"*-----------------------------------------------------------");
        Console.WriteLine($"*       Accuracy: {metrics?.Accuracy:P2}");
        Console.WriteLine($"*       Auc: {metrics?.AreaUnderRocCurve:P2}");
        Console.WriteLine($"*       Auprc: {metrics?.AreaUnderPrecisionRecallCurve:P2}");
        Console.WriteLine($"*       F1Score: {metrics?.F1Score:P2}");
        Console.WriteLine($"************************************************************");
    }
}