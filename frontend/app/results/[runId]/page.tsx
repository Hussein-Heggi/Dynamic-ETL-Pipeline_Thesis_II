'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { CheckCircle, XCircle, Loader2, Download } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import pipelineApi from '@/lib/api/pipeline';
import { PipelineWebSocket, WebSocketMessage } from '@/lib/websocket';

export default function ResultsPage() {
  const params = useParams();
  const runId = params.runId as string;

  const [wsMessages, setWsMessages] = useState<WebSocketMessage[]>([]);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState('pending');
  const [wsConnected, setWsConnected] = useState(false);

  // Fetch status
  const { data: status } = useQuery({
    queryKey: ['pipeline-status', runId],
    queryFn: () => pipelineApi.getStatus(runId),
    refetchInterval: (data) => {
      // Stop polling if completed or failed
      return data?.status === 'completed' || data?.status === 'failed' ? false : 2000;
    },
  });

  // Fetch results only when completed
  const { data: results } = useQuery({
    queryKey: ['pipeline-results', runId],
    queryFn: () => pipelineApi.getResults(runId),
    enabled: status?.status === 'completed',
  });

  // WebSocket connection
  useEffect(() => {
    let ws: PipelineWebSocket | null = null;

    if (runId && status?.status !== 'completed' && status?.status !== 'failed') {
      ws = new PipelineWebSocket(
        runId,
        (message) => {
          setWsMessages((prev) => [...prev, message]);
          if (message.progress !== undefined) {
            setCurrentProgress(message.progress);
          }
          if (message.stage) {
            setCurrentStage(message.stage);
          }
        },
        (error) => console.error('WebSocket error:', error),
        () => setWsConnected(false)
      );

      ws.connect();
      setWsConnected(true);
    }

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [runId, status?.status]);

  const getStageStatus = (stage: string) => {
    const stages = ['pending', 'ingestion', 'validation', 'transformation', 'completed'];
    const currentIndex = stages.indexOf(currentStage);
    const stageIndex = stages.indexOf(stage);

    if (status?.status === 'failed') {
      if (stageIndex <= currentIndex) return 'failed';
      return 'pending';
    }

    if (stageIndex < currentIndex) return 'completed';
    if (stageIndex === currentIndex) return 'active';
    return 'pending';
  };

  const StageIndicator = ({ stage, label }: { stage: string; label: string }) => {
    const stageStatus = getStageStatus(stage);

    return (
      <div className="flex flex-col items-center space-y-2">
        <div
          className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
            stageStatus === 'completed'
              ? 'bg-green-600'
              : stageStatus === 'active'
              ? 'bg-blue-600 animate-pulse'
              : stageStatus === 'failed'
              ? 'bg-red-600'
              : 'bg-gray-300'
          }`}
        >
          {stageStatus === 'completed' ? (
            <CheckCircle className="h-6 w-6 text-white" />
          ) : stageStatus === 'failed' ? (
            <XCircle className="h-6 w-6 text-white" />
          ) : stageStatus === 'active' ? (
            <Loader2 className="h-6 w-6 text-white animate-spin" />
          ) : (
            <div className="w-3 h-3 bg-white rounded-full" />
          )}
        </div>
        <span
          className={`text-sm font-medium ${
            stageStatus === 'active' ? 'text-blue-600' : 'text-gray-600'
          }`}
        >
          {label}
        </span>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-gray-900">Pipeline Execution</h1>
          {status && (
            <Badge
              variant={
                status.status === 'completed'
                  ? 'default'
                  : status.status === 'failed'
                  ? 'destructive'
                  : 'secondary'
              }
            >
              {status.status}
            </Badge>
          )}
        </div>
        <p className="text-gray-600">Run ID: {runId}</p>
      </div>

      {/* Progress Pipeline */}
      <Card>
        <CardHeader>
          <CardTitle>Pipeline Progress</CardTitle>
          <CardDescription>Real-time status of your ETL pipeline</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Stage Indicators */}
            <div className="flex items-center justify-between relative">
              <div className="absolute top-6 left-0 right-0 h-0.5 bg-gray-300 -z-10" />
              <StageIndicator stage="ingestion" label="Ingestion" />
              <StageIndicator stage="validation" label="Validation" />
              <StageIndicator stage="transformation" label="Transformation" />
              <StageIndicator stage="completed" label="Completed" />
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">{status?.message || 'Initializing...'}</span>
                <span className="font-medium">{currentProgress}%</span>
              </div>
              <Progress value={currentProgress} className="h-2" />
            </div>

            {/* Error Message */}
            {status?.error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm font-medium text-red-800">Error:</p>
                <p className="text-sm text-red-700 mt-1">{status.error}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Live Logs */}
      {wsConnected && wsMessages.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Live Logs</CardTitle>
            <CardDescription>Real-time updates from the pipeline</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto font-mono text-sm bg-gray-900 text-gray-100 p-4 rounded">
              {wsMessages.map((msg, idx) => (
                <div key={idx} className="flex items-start space-x-2">
                  <span className="text-gray-500">
                    [{new Date(msg.timestamp).toLocaleTimeString()}]
                  </span>
                  <span className={msg.type === 'error' ? 'text-red-400' : 'text-green-400'}>
                    {msg.message}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {results && (
        <Tabs defaultValue="data" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="data">Data Preview</TabsTrigger>
            <TabsTrigger value="validation">Validation Report</TabsTrigger>
            <TabsTrigger value="transformation">Transformation Report</TabsTrigger>
            <TabsTrigger value="downloads">Downloads</TabsTrigger>
          </TabsList>

          <TabsContent value="data" className="space-y-4">
            {results.dataframes.map((df, idx) => (
              <Card key={idx}>
                <CardHeader>
                  <CardTitle>DataFrame {idx}</CardTitle>
                  <CardDescription>
                    Shape: {df.shape[0]} rows × {df.shape[1]} columns
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm border-collapse">
                      <thead>
                        <tr className="border-b bg-gray-50">
                          {df.columns.map((col) => (
                            <th key={col} className="p-2 text-left font-medium">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {df.preview.map((row, rowIdx) => (
                          <tr key={rowIdx} className="border-b">
                            {df.columns.map((col) => (
                              <td key={col} className="p-2">
                                {String(row[col] ?? '')}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          <TabsContent value="validation">
            <Card>
              <CardHeader>
                <CardTitle>Validation Report</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="bg-gray-900 text-gray-100 p-4 rounded overflow-x-auto text-sm">
                  {JSON.stringify(results.validation_report, null, 2)}
                </pre>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="transformation">
            <Card>
              <CardHeader>
                <CardTitle>Transformation Report</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="bg-gray-900 text-gray-100 p-4 rounded overflow-x-auto text-sm">
                  {JSON.stringify(results.transformation_report, null, 2)}
                </pre>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="downloads">
            <Card>
              <CardHeader>
                <CardTitle>Download Files</CardTitle>
                <CardDescription>Download processed data and reports</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {results.dataframes.map((df, idx) => (
                    <a
                      key={idx}
                      href={pipelineApi.downloadFile(runId, `df_${idx}.csv`)}
                      download
                    >
                      <Button variant="outline" className="w-full justify-start">
                        <Download className="h-4 w-4 mr-2" />
                        Download DataFrame {idx} (CSV)
                      </Button>
                    </a>
                  ))}
                  <a
                    href={pipelineApi.downloadFile(runId, 'validation_report.json')}
                    download
                  >
                    <Button variant="outline" className="w-full justify-start">
                      <Download className="h-4 w-4 mr-2" />
                      Download Validation Report (JSON)
                    </Button>
                  </a>
                  <a
                    href={pipelineApi.downloadFile(runId, 'transformation_report.json')}
                    download
                  >
                    <Button variant="outline" className="w-full justify-start">
                      <Download className="h-4 w-4 mr-2" />
                      Download Transformation Report (JSON)
                    </Button>
                  </a>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
