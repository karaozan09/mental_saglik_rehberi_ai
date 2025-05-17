<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;

class BDTService
{
    protected $apiUrl;

    public function __construct()
    {
        $this->apiUrl = env('BDT_API_URL', 'http://localhost:5000/api');
    }

    public function analyzeText($text)
    {
        try {
            $response = Http::post($this->apiUrl . '/analyze', [
                'text' => $text
            ]);

            if ($response->successful()) {
                return $response->json()['data'];
            }

            throw new \Exception('BDT API yanıt vermedi: ' . $response->body());
        } catch (\Exception $e) {
            throw new \Exception('BDT API hatası: ' . $e->getMessage());
        }
    }

    public function checkHealth()
    {
        try {
            $response = Http::get($this->apiUrl . '/health');
            return $response->successful();
        } catch (\Exception $e) {
            return false;
        }
    }
} 