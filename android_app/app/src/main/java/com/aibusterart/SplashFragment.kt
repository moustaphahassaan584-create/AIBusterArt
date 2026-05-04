package com.aibusterart

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.navigation.fragment.findNavController
import com.aibusterart.R

class SplashFragment : Fragment() {

    private lateinit var noInternetMessage: TextView
    private lateinit var connectivityManager: ConnectivityManager
    private val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            super.onAvailable(network)
            // When internet becomes available, re-run the check on the main thread
            Handler(Looper.getMainLooper()).post {
                checkInternetAndProceed()
            }
        }
    }

    // Inflates the layout for the splash screen
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_splash, container, false)
    }

    // Initializes views and sets up network monitoring after the view is created
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        noInternetMessage = view.findViewById(R.id.no_internet_message)
        connectivityManager = requireContext().getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

        // Create a request to monitor internet availability
        val networkRequest = NetworkRequest.Builder()
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .build()
        // Register the callback so we know when the user goes online
        connectivityManager.registerNetworkCallback(networkRequest, networkCallback)

        // Perform the initial internet check
        checkInternetAndProceed()
    }

    // Cleans up resources when the fragment's view is destroyed
    override fun onDestroyView() {
        super.onDestroyView()
        // Unregister the callback to avoid memory leaks
        connectivityManager.unregisterNetworkCallback(networkCallback)
    }

    // Checks for internet connectivity and navigates to the upload screen if available
    private fun checkInternetAndProceed() {
        // If the device is online, hide the error and proceed to the next screen
        if (isNetworkAvailable()) {
            noInternetMessage.visibility = View.GONE
            // Add a 3-second delay to show the splash screen branding
            Handler(Looper.getMainLooper()).postDelayed({
                // Ensure the fragment is still active before performing navigation
                if (isAdded) {
                    findNavController().navigate(R.id.action_splashFragment_to_uploadFragment)
                }
            }, 3000)
        } else {
            // If offline, show the message asking the user to connect
            noInternetMessage.visibility = View.VISIBLE
        }
    }

    // Determines if the device currently has an active internet connection
    private fun isNetworkAvailable(): Boolean {
        // Get the current active network; return false if none exists
        val network = connectivityManager.activeNetwork ?: return false
        // Get capabilities for the active network; return false if unavailable
        val activeNetwork = connectivityManager.getNetworkCapabilities(network) ?: return false
        // Return true if the network is either Wi-Fi or Cellular
        return when {
            activeNetwork.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> true
            activeNetwork.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> true
            else -> false
        }
    }
}
