'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useMutation } from '@tanstack/react-query';
import { Sparkles, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import pipelineApi from '@/lib/api/pipeline';

const exampleQueries = [
  'Show me Apple stock prices from last 30 days with SMA 10 days',
  'Get Tesla stock data for the past 3 months with 20-day moving average',
  'Fetch Microsoft closing prices for last week with volume data',
  'Show me Google stock with RSI indicator for the past month',
];

const qualityProfiles = [
  {
    id: 'high_quality',
    label: 'High Quality',
    description: 'Strictest thresholds, fewer but highly accurate matches.',
  },
  {
    id: 'balanced',
    label: 'Balanced',
    description: 'Default mix of accuracy and coverage for most use cases.',
  },
  {
    id: 'high_volume',
    label: 'High Volume',
    description: 'Looser thresholds to maximize joins/unions on large datasets.',
  },
];

export default function QueryPage() {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [qualityProfile, setQualityProfile] = useState('balanced');

  const mutation = useMutation({
    mutationFn: (payload: { query: string; profile: string }) =>
      pipelineApi.runPipeline({
        query: payload.query,
        options: { quality_profile: payload.profile },
      }),
    onSuccess: (data) => {
      router.push(`/results/${data.run_id}`);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      mutation.mutate({ query, profile: qualityProfile });
    }
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-gray-900">Create New Query</h1>
        <p className="text-gray-600">
          Describe what stock data you need in natural language
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Sparkles className="h-5 w-5 text-blue-600" />
            <span>Your Query</span>
          </CardTitle>
          <CardDescription>
            Enter your data requirements in plain English. Our LLM will understand and
            process your request.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Show me Apple stock prices from last 30 days with SMA 10 days"
              className="min-h-[120px] text-base"
              disabled={mutation.isPending}
            />
            <div className="space-y-3">
              <p className="text-sm font-medium text-gray-700">Validation profile</p>
              <div className="space-y-2">
                {qualityProfiles.map((profile) => (
                  <label
                    key={profile.id}
                    className={`flex items-start space-x-3 border rounded-lg p-3 cursor-pointer transition ${
                      qualityProfile === profile.id ? 'border-blue-500 bg-blue-50' : 'hover:border-blue-300'
                    }`}
                  >
                    <input
                      type="checkbox"
                      className="mt-1 h-4 w-4 rounded border-gray-300"
                      checked={qualityProfile === profile.id}
                      onChange={() => setQualityProfile(profile.id)}
                      disabled={mutation.isPending}
                    />
                    <div>
                      <p className="font-medium text-gray-900">{profile.label}</p>
                      <p className="text-sm text-gray-600">{profile.description}</p>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            <div className="flex justify-between items-center">
              <p className="text-sm text-gray-500">
                {query.length} characters
              </p>
              <Button
                type="submit"
                size="lg"
                disabled={!query.trim() || mutation.isPending}
                className="min-w-[150px]"
              >
                {mutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Starting...
                  </>
                ) : (
                  'Run Pipeline'
                )}
              </Button>
            </div>

            {mutation.isError && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-800">
                  Error: {mutation.error instanceof Error ? mutation.error.message : 'Failed to start pipeline'}
                </p>
              </div>
            )}
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Example Queries</CardTitle>
          <CardDescription>Click any example to use it as your query</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {exampleQueries.map((example, index) => (
              <button
                key={index}
                onClick={() => handleExampleClick(example)}
                className="p-4 text-left border rounded-lg hover:bg-gray-50 hover:border-blue-300 transition-colors"
                disabled={mutation.isPending}
              >
                <p className="text-sm text-gray-700">{example}</p>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-blue-50 border-blue-200">
        <CardHeader>
          <CardTitle className="text-lg">Tips for Better Queries</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-gray-700">
            <li className="flex items-start">
              <span className="text-blue-600 mr-2">•</span>
              <span>Specify the stock ticker or company name</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 mr-2">•</span>
              <span>Include a time range (e.g., last 30 days, past 3 months)</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 mr-2">•</span>
              <span>Mention any indicators you want (SMA, EMA, RSI, etc.)</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 mr-2">•</span>
              <span>Be as specific as possible for better results</span>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
