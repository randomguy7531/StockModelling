namespace StockLearning.FSharp

module DataHelpers =
    open DataPoints
    open System.IO

    let getFileLines fileName =
        let stream = new StreamReader(path = fileName)
        seq
            {
            while not stream.EndOfStream do
            yield stream.ReadLine()
            }

    let readTickerFile fileName =
        let tickerFileContents = getFileLines fileName
        let tickerLine = tickerFileContents |> Seq.head
        let splitters = [' '] |> Seq.toArray
        tickerLine.Split(splitters) |> Seq.toList

    let readStockDataFileToCurve fileDir ticker =
        getFileLines(fileDir + ticker + ".csv") 
        |> Seq.map(fun element -> StockPoint.FromCommaDelimitedString(element + "," + ticker)) 
        |> Seq.filter(fun element -> element.IsSome) 
        |> Seq.map(fun element -> element.Value)
        |> Seq.toList 
        |> StockCurve.FromStockPoints

    let readStockDataFilesFromTickers filesDir tickers = 
        let result = tickers |> Seq.map(fun element -> readStockDataFileToCurve filesDir element) |> Seq.toList
        result


        