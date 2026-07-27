plugins {
    alias(libs.plugins.android.library)
    id("maven-publish")
}

android {
    namespace = "com.yolomnn"
    compileSdk = 36

    defaultConfig {
        minSdk = 26
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        buildConfig = false
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
}

dependencies {
    testImplementation(libs.junit)
}

kotlin {
    jvmToolchain(17)
}

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "com.yolomnn"
            artifactId = "yolo-mnn"
            version = "1.0.0"
            afterEvaluate {
                from(components["release"])
            }
            pom {
                name.set("yolo-mnn")
                description.set("YOLO11 (n/s) on-device object detection for Android, powered by the MNN inference engine")
                url.set("https://github.com/xxddccaa/yolo11-mnn-android")
                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                    }
                }
                scm {
                    url.set("https://github.com/xxddccaa/yolo11-mnn-android")
                }
            }
        }
    }
}
