namespace StockLearning.FSharp

module FeatureBuilders = 
    open System
    open DataPoints

    let CurrentPointMetricFeature (dataIn : StockLearningData) (predictionPoint : StockPoint) (metric : DataPoints.PointMetrics)= 
        match metric with
        | PointMetrics.Open -> predictionPoint.Open
        | PointMetrics.Close -> predictionPoint.Close
        | PointMetrics.High -> predictionPoint.High
        | PointMetrics.Low -> predictionPoint.Low
        | PointMetrics.Volume -> (double)predictionPoint.Volume

    let ImmediatelyPriorPointMetricFeature (dataIn : StockLearningData) (predictionPoint : StockPoint) (metric : DataPoints.PointMetrics)= 
        let immediatelyPrevPoints = 
            dataIn.stockData 
            |> Seq.filter(fun curve -> curve.Symbol = predictionPoint.Symbol)
            |> Seq.collect(fun curve -> DataHelpers.getPriorPoints curve predictionPoint.Datetime)
        let immediatelyPrevPoint =
            immediatelyPrevPoints |> Seq.maxBy(fun point -> point.Datetime)
        match metric with
        | PointMetrics.Open -> immediatelyPrevPoint.Open
        | PointMetrics.Close -> immediatelyPrevPoint.Close
        | PointMetrics.High -> immediatelyPrevPoint.High
        | PointMetrics.Low -> immediatelyPrevPoint.Low
        | PointMetrics.Volume -> (double)immediatelyPrevPoint.Volume
