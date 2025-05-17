<?php

namespace App\Http\Controllers;

use App\Services\BDTService;
use Illuminate\Http\Request;

class BDTController extends Controller
{
    protected $bdtService;

    public function __construct(BDTService $bdtService)
    {
        $this->bdtService = $bdtService;
    }

    public function analyze(Request $request)
    {
        $request->validate([
            'text' => 'required|string'
        ]);

        try {
            $result = $this->bdtService->analyzeText($request->text);
            return response()->json([
                'success' => true,
                'data' => $result
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => $e->getMessage()
            ], 500);
        }
    }

    public function checkHealth()
    {
        $isHealthy = $this->bdtService->checkHealth();
        return response()->json([
            'success' => true,
            'healthy' => $isHealthy
        ]);
    }
} 