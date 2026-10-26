Public Sub httpclient()

Dim apiurl As String
Dim xmlhttp As New WinHttpRequest

'use default port of http requests
apiurl = "http://127.0.0.1:8088/"
endpoint = "api/"
params = "TEXT="
'tickers = "['Technology has journeyed from hardware to enterprise software to SMAC (social media, analytics and cloud) and artificial intelligence, while at the same time; it has become an integral part of every industry in an increasingly multi-device connected world. Smart machines, cognitive computing and internet of things are narrowing divide between humans and machines, and creating new applications in a world of cognizant computing where technology will take decisions for humans based on historical data and preferences. Thus, Information Technology is continuously evolving and the trend is expected to continue in future.']"
'tickers = "['As informed, the Group has competitive edge over peers due to its robust customer support service, ease in using software modules and value-added functionalities like analytics built-in the software which to some extent ensures the client stickiness along with long established relations with the client with about 90% of business coming from repeat orders..']"
'tickers = "['Increasing thrust on improving efficiencies to drive growth of the Group.']"
'tickers = "['Manpower attrition risk along with increasing salary cost.']"
'tickers = "['As informed, the Group has a strategy and action in place for advance hiring for attrition based on projected attrition of each department. This is for minimizing the impact of higher attrition in important functions like product development and delivery.']"
'tickers = "['The Group operates in more than 20 countries and derives around 99% of its revenue from international markets. This exposes the group to foreign exchange fluctuation risk. Considering its major exposure to the international market and the current volatility in the foreign exchange market, undertaking proper hedging mechanism to mitigate the foreign exchange currency fluctuation risk becomes critical for Client group.']"
'tickers = "['With automobile industry consuming the highest volume of casting," & _
"the demand scenario in casting industry is directly linked to the automobile sector. " & _
"Indian automobile industry is currently during a slowdown, with demand growing " & _
"by slowest rate in over a decade.']"

requestString = apiurl & endpoint & "?" & params & tickers
'requestString = apiurl & endpoint

Debug.Print requestString

xmlhttp.Open "POST", requestString, False

With xmlhttp
.send
End With


MsgBox (xmlhttp.responseText)

xmlhttp.abort
End Sub
