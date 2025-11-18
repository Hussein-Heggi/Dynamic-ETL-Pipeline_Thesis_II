'use client';

import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import { formatDistanceToNow, format } from 'date-fns';
import { CheckCircle, XCircle, Clock, Eye, Trash2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import pipelineApi from '@/lib/api/pipeline';

export default function HistoryPage() {
  const { data: history, isLoading } = useQuery({
    queryKey: ['pipeline-history'],
    queryFn: () => pipelineApi.getHistory(),
  });

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
      ingestion: 'secondary',
      validation: 'secondary',
      transformation: 'secondary',
    };
    return variants[status] || 'outline';
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Pipeline History</h1>
          <p className="text-gray-600 mt-2">View all your pipeline executions</p>
        </div>
        <Link href="/query">
          <Button>New Query</Button>
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>All Pipeline Runs</CardTitle>
          <CardDescription>
            {history ? `${history.length} total runs` : 'Loading...'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              <p className="mt-4 text-gray-500">Loading history...</p>
            </div>
          ) : history && history.length > 0 ? (
            <div className="border rounded-lg overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Status</TableHead>
                    <TableHead>Query</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {history.map((item) => (
                    <TableRow key={item.run_id}>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(item.status)}
                          <Badge variant={getStatusBadge(item.status)}>
                            {item.status}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell className="max-w-md">
                        <p className="truncate font-medium">{item.query}</p>
                        <p className="text-xs text-gray-500 mt-1">
                          ID: {item.run_id.substring(0, 8)}...
                        </p>
                      </TableCell>
                      <TableCell>
                        <div className="space-y-1">
                          <p className="text-sm">
                            {format(new Date(item.created_at), 'MMM d, yyyy')}
                          </p>
                          <p className="text-xs text-gray-500">
                            {format(new Date(item.created_at), 'h:mm a')}
                          </p>
                          <p className="text-xs text-gray-400">
                            {formatDistanceToNow(new Date(item.created_at), {
                              addSuffix: true,
                            })}
                          </p>
                        </div>
                      </TableCell>
                      <TableCell>
                        {item.duration ? (
                          <span className="text-sm font-mono">
                            {item.duration.toFixed(1)}s
                          </span>
                        ) : (
                          <span className="text-sm text-gray-400">-</span>
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end space-x-2">
                          {item.status === 'completed' && (
                            <Link href={`/results/${item.run_id}`}>
                              <Button size="sm" variant="outline">
                                <Eye className="h-4 w-4 mr-1" />
                                View
                              </Button>
                            </Link>
                          )}
                          {item.status !== 'completed' && item.status !== 'failed' && (
                            <Link href={`/results/${item.run_id}`}>
                              <Button size="sm" variant="outline">
                                <Clock className="h-4 w-4 mr-1" />
                                Monitor
                              </Button>
                            </Link>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="text-center py-12 space-y-4">
              <div className="text-gray-400">
                <Clock className="h-16 w-16 mx-auto" />
              </div>
              <div>
                <p className="text-lg font-medium text-gray-900">No pipeline runs yet</p>
                <p className="text-gray-500 mt-1">
                  Create your first query to see it here
                </p>
              </div>
              <Link href="/query">
                <Button>Create Your First Query</Button>
              </Link>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Stats Summary */}
      {history && history.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Completed Runs</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-600">
                {history.filter((h) => h.status === 'completed').length}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Failed Runs</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-red-600">
                {history.filter((h) => h.status === 'failed').length}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Average Duration</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {history.filter((h) => h.duration).length > 0
                  ? (
                      history
                        .filter((h) => h.duration)
                        .reduce((acc, h) => acc + (h.duration || 0), 0) /
                      history.filter((h) => h.duration).length
                    ).toFixed(1)
                  : '0'}
                s
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
