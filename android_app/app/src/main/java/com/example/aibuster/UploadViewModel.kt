package com.aibusterart

import android.net.Uri
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aibusterart.BuildConfig
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.DefaultRequest
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

class UploadViewModel : ViewModel() {

    private val HUGGING_FACE_API_TOKEN = BuildConfig.HUGGING_FACE_API_TOKEN
    private val HUGGING_FACE_API_URL = "https://me6345333-ai-image-detector.hf.space/run/predict"

    private val _classificationResult = MutableLiveData<Result<Pair<String, Float>>>()
    val classificationResult: LiveData<Result<Pair<String, Float>>> = _classificationResult

    private val client = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                isLenient = true
                prettyPrint = true
            })
        }
        install(HttpTimeout) {
            requestTimeoutMillis = 30000
        }
        install(DefaultRequest) {
            header("Authorization", "Bearer $HUGGING_FACE_API_TOKEN")
            contentType(ContentType.Application.Json)
        }
    }

    // Sends the image to the AI model API and processes the classification result
    fun classifyImage(imageBytes: ByteArray) {
        viewModelScope.launch {
            try {
                // Encode the image to Base64 to be sent in the JSON request
                val base64Image = android.util.Base64.encodeToString(imageBytes, android.util.Base64.DEFAULT)
                val dataUri = "data:image/jpeg;base64,$base64Image"

                // Create the JSON request body following the API's expected format
                val requestBody = buildJsonObject {
                    put("data", buildJsonArray {
                        add(JsonPrimitive(dataUri))
                    })
                }

                // Execute the POST request to the Hugging Face space
                val httpResponse = client.post(HUGGING_FACE_API_URL) {
                    setBody(requestBody)
                }

                // Check if the API request was successful
                if (!httpResponse.status.isSuccess()) {
                    val errorBody = httpResponse.bodyAsText()
                    Log.e("UploadViewModel", "Server error: ${httpResponse.status}. Body: $errorBody")
                    _classificationResult.postValue(Result.failure(Exception("Failed to classify image. Status: ${httpResponse.status}")))
                    return@launch
                }

                // Parse the JSON response to extract classification data
                val response: JsonElement = httpResponse.body()
                val dataArray = response.jsonObject["data"]?.jsonArray

                // Ensure the response contains the expected data array
                if (dataArray != null && dataArray.isNotEmpty()) {
                    val predictionResult = dataArray[0].jsonObject
                    val topLabel = predictionResult["label"]?.jsonPrimitive?.content
                    val confidencesArray = predictionResult["confidences"]?.jsonArray

                    var artificialScore = 0.0
                    // Iterate through confidence scores to find the "artificial" probability
                    if (confidencesArray != null) {
                        val artificialConfidenceObject = confidencesArray.firstOrNull {
                            it.jsonObject["label"]?.jsonPrimitive?.content == "artificial"
                        }?.jsonObject

                        // Extract the confidence value if the "artificial" label is found
                        if (artificialConfidenceObject != null) {
                            artificialScore = artificialConfidenceObject["confidence"]?.jsonPrimitive?.doubleOrNull ?: 0.0
                        }
                    }

                    // Map the top label to a user-friendly string and post the result
                    if (topLabel != null) {
                        val resultLabel = if (topLabel == "artificial") "AI-Generated" else "Human-Generated"
                        _classificationResult.postValue(Result.success(resultLabel to artificialScore.toFloat()))
                    } else {
                        // Handle cases where the label is missing in the response
                        _classificationResult.postValue(Result.failure(Exception("Failed to parse prediction label.")))
                    }
                } else {
                    // Handle cases where the data array is empty or null
                    _classificationResult.postValue(Result.failure(Exception("Failed to get predictions from the server.")))
                }

            } catch (e: Exception) {
                // Catch and report any network or parsing exceptions
                Log.e("UploadViewModel", "Error classifying image", e)
                _classificationResult.postValue(Result.failure(e))
            }
        }
    }

    // Closes the network client when the ViewModel is cleared to free up resources
    override fun onCleared() {
        super.onCleared()
        client.close()
    }
}
