package com.aibusterart

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.os.bundleOf
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.navigation.fragment.findNavController
import com.aibusterart.BuildConfig
import com.aibusterart.R
import java.io.ByteArrayOutputStream
import java.io.InputStream

class UploadFragment : Fragment() {

    private lateinit var imagePickerLauncher: ActivityResultLauncher<Intent>
    private var selectedImageUri: Uri? = null

    private lateinit var imagePlaceholder: FrameLayout
    private lateinit var previewImage: ImageView
    private lateinit var classifyButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var uploadHint: LinearLayout

    private val viewModel: UploadViewModel by viewModels()

    // Inflates the layout for the upload screen
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_upload, container, false)
    }

    // Initializes UI components and sets up click listeners after the view is created
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        imagePlaceholder = view.findViewById(R.id.image_placeholder)
        previewImage = view.findViewById(R.id.preview_image)
        classifyButton = view.findViewById(R.id.classify_button)
        progressBar = view.findViewById(R.id.progress_bar)
        uploadHint = view.findViewById(R.id.upload_hint)
        val infoButton = view.findViewById<ImageView>(R.id.info_button)

        // Navigate to the About screen when the info button is clicked
        infoButton.setOnClickListener {
            findNavController().navigate(R.id.action_uploadFragment_to_aboutFragment)
        }

        // Define the logic for handling the selected image from the gallery
        imagePickerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            // If the user successfully picked an image
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    selectedImageUri = uri
                    // Update the UI to show the selected image and the classify button
                    previewImage.setImageURI(uri)
                    previewImage.visibility = View.VISIBLE
                    uploadHint.visibility = View.GONE
                    classifyButton.visibility = View.VISIBLE
                }
            }
        }

        // Open the system gallery when the image placeholder is clicked
        imagePlaceholder.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            imagePickerLauncher.launch(intent)
        }

        // Initiate classification when the classify button is clicked
        classifyButton.setOnClickListener {
            // Verify that the API token is configured before proceeding
            if (BuildConfig.HUGGING_FACE_API_TOKEN.isEmpty() || BuildConfig.HUGGING_FACE_API_TOKEN == "PASTE_YOUR_HUGGING_FACE_TOKEN_HERE") {
                Toast.makeText(context, "Please set your Hugging Face API token in local.properties", Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            // Ensure an image is selected before starting classification
            selectedImageUri?.let {
                classifyImage(it)
            }
        }

        // Observe the classification result from the ViewModel
        viewModel.classificationResult.observe(viewLifecycleOwner) {
            // Hide progress bar and re-enable button when a result (success or failure) is received
            progressBar.visibility = View.GONE
            classifyButton.isEnabled = true
            
            it.onSuccess {
                // If successful, extract data and navigate to the result screen
                val (resultLabel, confidence) = it
                val bundle = bundleOf(
                    "imageUri" to selectedImageUri.toString(),
                    "resultLabel" to resultLabel,
                    "resultConfidence" to confidence
                )
                findNavController().navigate(R.id.action_uploadFragment_to_resultFragment, bundle)
            }.onFailure {
                // If an error occurred, show a message to the user
                Toast.makeText(context, "Error: ${it.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    // Prepares the image and initiates the classification process via the ViewModel
    private fun classifyImage(uri: Uri) {
        // Show loading state to the user
        progressBar.visibility = View.VISIBLE
        classifyButton.isEnabled = false
        // Resize image to optimize network usage and processing
        val imageBytes = getResizedImageBytes(uri)
        viewModel.classifyImage(imageBytes)
    }

    // Resizes the selected image and converts it to a byte array for API upload
    private fun getResizedImageBytes(uri: Uri): ByteArray {
        val inputStream: InputStream? = requireContext().contentResolver.openInputStream(uri)

        // Decode only the dimensions of the image first
        val options = BitmapFactory.Options()
        options.inJustDecodeBounds = true
        BitmapFactory.decodeStream(inputStream, null, options)
        inputStream?.close()

        // Calculate the sampling size to resize the image to a maximum of 512x512
        var inSampleSize = 1
        val reqWidth = 512
        val reqHeight = 512
        while (options.outWidth / inSampleSize > reqWidth || options.outHeight / inSampleSize > reqHeight) {
            inSampleSize *= 2
        }

        // Decode the image with the calculated sampling size
        val scaleOptions = BitmapFactory.Options()
        scaleOptions.inSampleSize = inSampleSize

        val scaledInputStream = requireContext().contentResolver.openInputStream(uri)
        val scaledBitmap = BitmapFactory.decodeStream(scaledInputStream, null, scaleOptions)
        scaledInputStream?.close()

        // Compress the bitmap into a JPEG byte array
        val stream = ByteArrayOutputStream()
        scaledBitmap?.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
    }
}
