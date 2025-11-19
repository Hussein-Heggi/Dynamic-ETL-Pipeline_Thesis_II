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
import pipelineApi, { StageFlags } from '@/lib/api/pipeline';
import { PipelineWebSocket, WebSocketMessage } from '@/lib/websocket';

const EMPTY_STAGE_FLAGS: StageFlags = {
  ingestion: false,
  validation: false,
  transformation: false,
  completed: false,
};

const normalizeStageFlags = (flags?: Partial<StageFlags>): StageFlags => ({
  ...EMPTY_STAGE_FLAGS,
  ...flags,
});

type StageKey = keyof StageFlags;
const STAGE_SEQUENCE: StageKey[] = ['ingestion', 'validation', 'transformation', 'completed'];

export default function ResultsPage() {
  const params = useParams();
  const runId = params.runId as string;

  const [wsMessages, setWsMessages] = useState<WebSocketMessage[]>([]);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState('pending');
  const [stageFlags, setStageFlags] = useState<StageFlags>({ ...EMPTY_STAGE_FLAGS });
  const [wsConnected, setWsConnected] = useState(false);

  const { data: status } = useQuery({
    queryKey: ['pipeline-status', runId],
    queryFn: () => pipelineApi.getStatus(runId),
    refetchInterval: (data) => {
      return data?.status === 'completed' || data?.status === 'failed' ? false : 1000;
    },
  });

  const { data: results } = useQuery({
    queryKey: ['pipeline-results', runId],
    queryFn: () => pipelineApi.getResults(runId),
    enabled: status?.status === 'completed',
  });

useEffect(() => {
  setStageFlags({ ...EMPTY_STAGE_FLAGS });
}, [runId]);

useEffect(() => {
  if (status) {
    setCurrentProgress(status.progress);
    setCurrentStage(status.status);
    setStageFlags(normalizeStageFlags(status.stage_flags));
  }
}, [status]);

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
          const flagUpdate = message.stage_flags || message.data?.stage_flags;
          if (flagUpdate) {
            setStageFlags(normalizeStageFlags(flagUpdate));
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

  const getStageStatus = (stage: StageKey) => {
    if (stageFlags[stage]) {
      return 'completed';
    }

    const firstIncompleteIndex = STAGE_SEQUENCE.findIndex((name) => !stageFlags[name]);
    const stageIndex = STAGE_SEQUENCE.indexOf(stage);

    if (status?.status === 'failed') {
      if (firstIncompleteIndex === -1) {
        return 'completed';
      }
      if (stageIndex === firstIncompleteIndex) {
        return 'failed';
      }
      if (stageIndex < firstIncompleteIndex) {
        return 'completed';
      }
      return 'pending';
    }

    if (firstIncompleteIndex === -1) {
      return 'completed';
    }

    if (stageIndex === firstIncompleteIndex && status?.status !== 'pending') {
      return 'active';
    }

    if (stageIndex < firstIncompleteIndex) {
      return 'completed';
    }

    return 'pending';
  };

const StageIndicator = ({ stage, label }: { stage: StageKey; label: string }) => {
  const stageStatus = getStageStatus(stage);

  return (
      <div className="flex flex-col items-center space-y-2">
        <div
          className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 ${
            stageStatus === 'completed'
              ? 'bg-green-600 scale-100'
              : stageStatus === 'active'
              ? 'bg-blue-600 scale-110'
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
          className={`text-sm font-medium transition-colors ${
            stageStatus === 'active' ? 'text-blue-600 font-bold' :
            stageStatus === 'completed' ? 'text-green-600' :
            'text-gray-600'
          }`}
        >
          {label}
        </span>
      </div>
  );
};

const formatShape = (shape?: [number, number]) => {
  if (!shape || shape.length < 2) return 'Unknown shape';
  return `${shape[0]} rows × ${shape[1]} columns`;
};

const formatNullHandling = (nullHandling?: any) => {
  if (!nullHandling) return null;
  const deleted =
    nullHandling.columns_deleted?.length > 0
      ? `${nullHandling.columns_deleted.length} column(s) dropped (${nullHandling.columns_deleted
          .map((col: any) => `${col.column} ${(col.null_ratio * 100).toFixed(1)}%`)
          .join(', ')})`
      : 'No columns dropped';

  const imputedEntries = Object.entries(nullHandling.columns_imputed || {});
  const imputed =
    imputedEntries.length > 0
      ? `${imputedEntries.length} column(s) imputed (${imputedEntries
          .map(([col, info]: [string, any]) => `${col} (${info.method})`)
          .join(', ')})`
      : 'No imputation needed';

  return { deleted, imputed };
};

const formatWarnings = (warnings?: any) => {
  if (!warnings) return [];
  const entries: string[] = [];
  Object.entries(warnings).forEach(([key, value]) => {
    entries.push(`${key.replaceAll('_', ' ')}: ${value}`);
  });
  return entries;
};

const formatErrors = <T,>(errors?: T[]) =>
  errors && errors.length > 0 ? (
    <ul className="list-disc ml-5">
      {errors.map((err: any, idx: number) => (
        <li key={idx}>{typeof err === 'string' ? err : JSON.stringify(err)}</li>
      ))}
    </ul>
  ) : null;

const SummaryStat = ({ title, value, subtitle }: { title: string; value: string | number; subtitle?: string }) => (
  <div className="p-4 border rounded-lg bg-gray-50">
    <p className="text-sm font-medium text-gray-500">{title}</p>
    <p className="text-2xl font-semibold text-gray-900">{value}</p>
    {subtitle && <p className="text-sm text-gray-600 mt-1">{subtitle}</p>}
  </div>
);

const getJoinSummary = (value: any) => {
  if (!value) return 'Executed';
  if (Array.isArray(value)) {
    const count = value.length;
    return `${count} match group${count === 1 ? '' : 's'}`;
  }
  if (typeof value === 'object') {
    if (typeof value.summary === 'string') return value.summary;
    const keys = Object.keys(value);
    if (keys.length === 0) return 'Executed';
    return keys
      .map((key) => `${key.replaceAll('_', ' ')}: ${JSON.stringify(value[key])}`)
      .join(', ');
  }
  return String(value);
};

const ValidationSummary = ({ report }: { report: any }) => {
  const inputShapes = (report.input_shapes || []).map(formatShape).join(', ');
  const outputShapes = (report.output_shapes || []).map(formatShape).join(', ');
  const unionOps = report.union_operations || [];
  const joinEntries = Object.entries(report.join_operations || {}).filter(
    ([, value]) =>
      value &&
      ((Array.isArray(value) && value.length > 0) ||
        (typeof value === 'object' && Object.keys(value).length > 0))
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <SummaryStat
          title="Input Tables"
          value={report.input_count ?? '--'}
          subtitle={inputShapes ? `Shapes: ${inputShapes}` : undefined}
        />
        <SummaryStat
          title="Outputs"
          value={report.output_count ?? '--'}
          subtitle={outputShapes ? `Shapes: ${outputShapes}` : undefined}
        />
        <SummaryStat
          title="Union Ops"
          value={unionOps.length}
          subtitle={unionOps.length > 0 ? 'See details below' : 'No unions were needed'}
        />
      </div>

      <div className="flex flex-wrap gap-3">
        <Badge variant={report.early_termination ? 'default' : 'outline'}>
          Early termination: {report.early_termination ? 'Yes' : 'No'}
        </Badge>
        <Badge variant={report.stage_2_skipped ? 'outline' : 'default'}>
          Stage 2 skipped: {report.stage_2_skipped ? 'Yes' : 'No'}
        </Badge>
      </div>

      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-gray-800">Union Operations</h4>
        {unionOps.length > 0 ? (
          <div className="space-y-3">
            {unionOps.map((op: any, idx: number) => (
              <div key={idx} className="border rounded-lg p-3">
                <p className="font-medium text-gray-900">{op.group || `Union ${idx + 1}`}</p>
                <p className="text-sm text-gray-600">
                  Score: {typeof op.score === 'number' ? op.score.toFixed(2) : op.score ?? 'n/a'}
                </p>
                {op.result_shape && (
                  <p className="text-sm text-gray-600">Result shape: {formatShape(op.result_shape)}</p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-600">No unions performed in this run.</p>
        )}
      </div>

      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-gray-800">Join Operations</h4>
        {joinEntries.length > 0 ? (
          <div className="space-y-3">
            {joinEntries.map(([stage, value]) => (
              <div key={stage} className="border rounded-lg p-3">
                <p className="font-medium text-gray-900">{stage}</p>
                <p className="text-sm text-gray-600">{getJoinSummary(value)}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-600">No joins were executed.</p>
        )}
      </div>
    </div>
  );
};

const TransformationSummary = ({ report }: { report: any }) => {
  const results = report.results || [];
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <SummaryStat title="Dataframes Processed" value={report.dataframes_processed ?? results.length} />
        <SummaryStat title="Overall Status" value={report.overall_status ?? 'Unknown'} />
        <SummaryStat title="Total Errors" value={report.total_errors ?? 0} />
      </div>

      {results.length === 0 && <p className="text-sm text-gray-600">No transformation details are available.</p>}

      <div className="space-y-4">
        {results.map((result: any, idx: number) => (
          <div key={`${result.index}-${idx}`} className="border rounded-lg p-4 space-y-2">
            <div className="flex items-start justify-between">
              <div>
                <p className="font-semibold text-gray-900">Output Dataset {result.index}</p>
                <p className="text-sm text-gray-600">
                  {formatShape(result.original_shape)} → {formatShape(result.final_shape || result.cleaned_shape)}
                </p>
              </div>
              <Badge
                variant={
                  result.status === 'success'
                    ? 'default'
                    : result.status === 'failure'
                    ? 'destructive'
                    : 'secondary'
                }
              >
                {result.status || 'unknown'}
              </Badge>
            </div>
            {result.keywords?.length > 0 && (
              <p className="text-sm text-gray-700">
                Keywords: <span className="font-medium text-gray-900">{result.keywords.join(', ')}</span>
              </p>
            )}
            {result.cleaning && (
              <div className="text-sm text-gray-600 space-y-1">
                <p className="font-medium text-gray-800">Initial cleaning</p>
                {result.cleaning.missing_required_columns?.length > 0 && (
                  <p>Missing required columns: {result.cleaning.missing_required_columns.join(', ')}</p>
                )}
                {formatWarnings(result.cleaning.warnings).length > 0 && (
                  <div>
                    <p className="font-medium text-gray-700">Warnings</p>
                    <ul className="list-disc ml-5">
                      {formatWarnings(result.cleaning.warnings).map((warning, idx) => (
                        <li key={idx}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {result.cleaning.null_handling && (() => {
                  const info = formatNullHandling(result.cleaning.null_handling);
                  if (!info) return null;
                  return (
                    <>
                      <p>Null threshold: {result.cleaning.null_handling.threshold ?? 'n/a'}</p>
                      <p>{info.deleted}</p>
                      <p>{info.imputed}</p>
                    </>
                  );
                })()}
                {result.cleaning.pandera && (
                  <p>
                    Pandera validation:{' '}
                    <span className={result.cleaning.pandera.status?.includes('failed') ? 'text-red-600' : 'text-green-600'}>
                      {result.cleaning.pandera.status}
                    </span>
                  </p>
                )}
              </div>
            )}
            {result.enrichment && (
              <div className="text-sm space-y-1">
                <p className="font-medium text-gray-800">Enrichment</p>
                <p className={result.enrichment.success ? 'text-gray-600' : 'text-red-600'}>
                  Status: {result.enrichment.success ? 'Successful' : 'Failed'}
                </p>
                {result.enrichment.dsl_string && (
                  <pre className="bg-gray-900 text-gray-100 rounded p-2 text-xs overflow-x-auto">
                    {result.enrichment.dsl_string}
                  </pre>
                )}
                {result.enrichment.errors?.length > 0 && (
                  <div className="text-red-600">
                    <p className="font-medium">Errors</p>
                    {formatErrors(result.enrichment.errors)}
                  </div>
                )}
              </div>
            )}
            {result.errors?.length > 0 && (
              <div className="text-sm text-red-600">
                <p className="font-medium text-red-700">Pipeline errors</p>
                {formatErrors(result.errors)}
              </div>
            )}
            {result.post_enrichment_cleaning && (
              <div className="text-sm text-gray-600 space-y-1">
                <p className="font-medium text-gray-800">Post-enrichment cleaning</p>
                {result.post_enrichment_cleaning.null_handling && (() => {
                  const info = formatNullHandling(result.post_enrichment_cleaning.null_handling);
                  if (!info) return null;
                  return (
                    <>
                      <p>Null threshold: {result.post_enrichment_cleaning.null_handling.threshold ?? 'n/a'}</p>
                      <p>{info.deleted}</p>
                      <p>{info.imputed}</p>
                    </>
                  );
                })()}
                {result.post_enrichment_cleaning.pandera && (
                  <p>
                    Pandera validation:{' '}
                    <span
                      className={
                        result.post_enrichment_cleaning.pandera.status?.includes('failed')
                          ? 'text-red-600'
                          : 'text-green-600'
                      }
                    >
                      {result.post_enrichment_cleaning.pandera.status}
                    </span>
                  </p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-gray-900">Pipeline Execution</h1>
          {status && status.status !== 'pending' && status.status !== 'ingestion' && status.status !== 'validation' && status.status !== 'transformation' && (
            <Badge
              variant={
                status.status === 'completed'
                  ? 'default'
                  : status.status === 'failed'
                  ? 'destructive'
                  : 'secondary'
              }
              className="text-base px-4 py-1"
            >
              {status.status}
            </Badge>
          )}
        </div>
        {status?.query && (
          <p className="text-gray-600">
            <span className="font-semibold text-gray-700">Prompt:</span>{' '}
            <span>{status.query}</span>
          </p>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Pipeline Progress</CardTitle>
          <CardDescription>Real-time status of your ETL pipeline</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="flex items-center justify-between relative">
              <div className="absolute top-6 left-0 right-0 h-1 bg-gray-200 -z-10">
                <div
                  className="h-full bg-blue-600 transition-all duration-500"
                  style={{
                    width: `${Math.min(currentProgress, 100)}%`,
                  }}
                />
              </div>

              <StageIndicator stage="ingestion" label="Ingestion" />
              <StageIndicator stage="validation" label="Validation" />
              <StageIndicator stage="transformation" label="Transformation" />
              <StageIndicator stage="completed" label="Completed" />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center text-sm">
                <div className="flex items-center space-x-2">
                  <span className="text-gray-600 font-medium">
                    {status?.message || 'Initializing...'}
                  </span>
                </div>
                <span className="font-bold text-lg">{Math.round(currentProgress)}%</span>
              </div>
              <Progress value={currentProgress} className="h-3" />
            </div>

            {status?.error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm font-medium text-red-800">Error:</p>
                <p className="text-sm text-red-700 mt-1">{status.error}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {wsConnected && wsMessages.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <span>Live Logs</span>
              <Badge variant="outline" className="animate-pulse">Live</Badge>
            </CardTitle>
            <CardDescription>Real-time updates from the pipeline</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto font-mono text-sm bg-gray-900 text-gray-100 p-4 rounded">
              {wsMessages.map((msg, idx) => (
                <div key={idx} className="flex items-start space-x-2">
                  <span className="text-gray-500 flex-shrink-0">
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
                <CardTitle>Validation Insights</CardTitle>
                <CardDescription>Key takeaways from the union/join stages</CardDescription>
              </CardHeader>
              <CardContent>
                {results.validation_report ? (
                  <ValidationSummary report={results.validation_report} />
                ) : (
                  <p className="text-sm text-gray-600">Validation report is not available yet.</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="transformation">
            <Card>
              <CardHeader>
                <CardTitle>Transformation Insights</CardTitle>
                <CardDescription>Cleaning and enrichment outcomes for each dataframe</CardDescription>
              </CardHeader>
              <CardContent>
                {results.transformation_report ? (
                  <TransformationSummary report={results.transformation_report} />
                ) : (
                  <p className="text-sm text-gray-600">Transformation report is not available yet.</p>
                )}
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
                        Download Output {idx} (CSV)
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
