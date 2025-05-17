<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\BDTController;

// BDT API Routes
Route::prefix('bdt')->group(function () {
    Route::get('/health', [BDTController::class, 'checkHealth']);
    Route::post('/analyze', [BDTController::class, 'analyzeText']);
}); 