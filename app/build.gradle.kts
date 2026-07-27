plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.yolomnn.demo"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.yolomnn.demo"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        viewBinding = true
    }

    // The .mnn models are large and already compressed; keep them uncompressed
    // so we can memory-map / copy them out of assets efficiently.
    androidResources {
        noCompress += "mnn"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":yolo"))
    implementation(libs.google.material)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)
    implementation(libs.kotlinx.coroutines.android)
}

kotlin {
    jvmToolchain(17)
}
