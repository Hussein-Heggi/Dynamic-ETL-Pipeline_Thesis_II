'use client';

import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import { ArrowRight, TrendingUp, CheckCircle, XCircle, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import pipelineApi from '@/lib/api/pipeline';
import { formatDistanceToNow } from 'date-fns';

export default function Home() {
  const { data: history, isLoading } = useQuery({
    queryKey: ['pipeline-history'],
    queryFn: () => pipelineApi.getHistory(),
  });

  // Calculate stats
  const totalRuns = history?.length || 0;
  const completedRuns = history?.filter(h => h.status === 'completed').length || 0;
  const failedRuns = history?.filter(h => h.status === 'failed').length || 0;
  const successRate = totalRuns > 0 ? Math.round((completedRuns / totalRuns) * 100) : 0;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-600" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
      completed: 'default',
      failed: 'destructive',
      pending: 'secondary',
    };
    return variants[status] || 'outline';
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gray-900">
          Welcome to Dynamic ETL Pipeline
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Transform natural language queries into processed stock market data using our
          LLM-powered pipeline. Start by creating a new query below.
        </p>
        <div className="flex justify-center gap-4 pt-4">
          <Link href="/query">
            <Button size="lg" className="flex items-center space-x-2">
              <span>Create New Query</span>
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
          <Link href="/history">
            <Button size="lg" variant="outline">
              View History
            </Button>
          </Link>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Runs</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{totalRuns}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Completed</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-600">{completedRuns}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Failed</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-red-600">{failedRuns}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Success Rate</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline space-x-2">
              <div className="text-3xl font-bold">{successRate}%</div>
              {successRate >= 80 && <TrendingUp className="h-5 w-5 text-green-600" />}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Pipeline Runs</CardTitle>
          <CardDescription>Your most recent ETL pipeline executions</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-center py-8 text-gray-500">Loading...</div>
          ) : history && history.length > 0 ? (
            <div className="space-y-4">
              {history.slice(0, 5).map((item) => (
                <div
                  key={item.run_id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-4 flex-1">
                    {getStatusIcon(item.status)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {item.query}
                      </p>
                      <p className="text-xs text-gray-500">
                        {formatDistanceToNow(new Date(item.created_at), { addSuffix: true })}
                        {item.duration && ` • ${item.duration.toFixed(1)}s`}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Badge variant={getStatusBadge(item.status)}>
                      {item.status}
                    </Badge>
                    {item.status === 'completed' && (
                      <Link href={`/results/${item.run_id}`}>
                        <Button size="sm" variant="outline">
                          View Results
                        </Button>
                      </Link>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 space-y-4">
              <p className="text-gray-500">No pipeline runs yet</p>
              <Link href="/query">
                <Button>Create Your First Query</Button>
              </Link>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Natural Language Queries</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">
              Simply describe what data you need in plain English. Our LLM understands your
              requirements and fetches the right data.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Real-time Processing</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">
              Watch your data flow through ingestion, validation, and transformation stages
              with live progress updates.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Comprehensive Reports</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">
              Get detailed validation and transformation reports, complete with data quality
              metrics and enrichment features.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
