package com.aibusterart

import android.net.Uri
import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.navigation.fragment.findNavController
import com.aibusterart.R

class ResultFragment : Fragment() {

    // Inflates the layout for the result screen
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_result, container, false)
    }

    // Displays the classification results and sets up the retry button after the view is created
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val resultImage = view.findViewById<ImageView>(R.id.result_image)
        val resultText = view.findViewById<TextView>(R.id.result_text)
        val aiConfidence = view.findViewById<TextView>(R.id.ai_confidence)
        val classifyAnotherButton = view.findViewById<Button>(R.id.classify_another_button)

        arguments?.let {
            val imageUriString = it.getString("imageUri")
            val resultLabel = it.getString("resultLabel")
            val resultConfidence = it.getFloat("resultConfidence")

            if (imageUriString != null) {
                resultImage.setImageURI(Uri.parse(imageUriString))
            }
            resultText.text = "Result: $resultLabel"
            aiConfidence.text = String.format("AI Confidence: %.1f%%", resultConfidence * 100)
        }

        classifyAnotherButton.setOnClickListener {
            findNavController().navigate(R.id.action_resultFragment_to_uploadFragment)
        }
    }
}
