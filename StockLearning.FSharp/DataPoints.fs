namespace StockLearning.FSharp

module DataPoints =
    open System

    type StockPoint = 
        {
            Symbol : string;
            Datetime : DateTime;
            Open : double;
            High : double;
            Low : double;
            Close : double;
            Volume : int64
        }

        static member FromCommaDelimitedString (inputString : System.String) = 
            let separators = [','] |> Seq.toArray
            let elements = inputString.Split(separators)
            let requiredElementsLength = 7

            if(elements.Length = requiredElementsLength) then
                let symString   = elements.[0]
                let dateString  = elements.[1]
                let openString  = elements.[2]
                let highString  = elements.[3]
                let lowString   = elements.[4]
                let closeString = elements.[5]
                let volString   = elements.[6]

                try
                    let parsedDT    = DateTime.Parse(dateString)
                    let parsedOpen  = Double.Parse(openString)
                    let parsedHigh  = Double.Parse(highString)
                    let parsedLow   = Double.Parse(lowString)
                    let parsedClose = Double.Parse(closeString)
                    let parsedVol   = LanguagePrimitives.ParseInt64(volString)

                    Some({Symbol = symString; Datetime = parsedDT; Open = parsedOpen; High = parsedHigh; Low = parsedLow; Close = parsedClose; Volume = parsedVol})
                with
                    | _ -> None
            else
                None

    type StockCurve = 
        {Points : List<StockPoint>}

        static member FromStockPoints (inputPoints : List<StockPoint>)= 
            {Points = inputPoints}