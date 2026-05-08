import java.util.Properties
import java.io.FileInputStream

plugins {
    alias(libs.plugins.android.application)
    id("org.jetbrains.kotlin.plugin.serialization") version "1.9.21"
}

// Read local.properties
val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localProperties.load(FileInputStream(localPropertiesFile))
}

android {
    namespace = "com.aibusterart"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.aibusterart"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Read the token and expose it as a build variable
        val replicateApiToken = localProperties.getProperty("replicate.api.token") ?: ""
        buildConfigField("String", "REPLICATE_API_TOKEN", "\"$replicateApiToken\"")
        val huggingFaceApiToken = localProperties.getProperty("huggingface.api.token") ?: ""
        buildConfigField("String", "HUGGING_FACE_API_TOKEN", "\"$huggingFaceApiToken\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        buildConfig = true
    }
}

dependencies {
    // Ktor for networking
    implementation("io.ktor:ktor-client-core:2.3.10")
    implementation("io.ktor:ktor-client-cio:2.3.10")
    implementation("io.ktor:ktor-client-content-negotiation:2.3.10")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.10")

    // Kotlinx Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")

    // ViewModel
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.0")
    // Fragment KTX for viewModels delegate
    implementation("androidx.fragment:fragment-ktx:1.7.0")

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.navigation.fragment.ktx)
    implementation(libs.androidx.navigation.ui.ktx)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}