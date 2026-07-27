package com.yolomnn.demo

import android.content.Context
import java.io.File

/** The two bundled model variants. Files live in app assets and are copied to
 *  cache on first use so the native MNN Interpreter can open them by path. */
enum class ModelVariant(val displayName: String, val assetName: String) {
    N("YOLO11n", "yolo11n.mnn"),
    S("YOLO11s", "yolo11s.mnn"),
}

object ModelAssets {
    /** Copy the model out of assets into cacheDir (once) and return its path. */
    fun ensureExtracted(context: Context, variant: ModelVariant): String {
        val out = File(context.cacheDir, variant.assetName)
        if (!out.exists() || out.length() == 0L) {
            context.assets.open(variant.assetName).use { input ->
                out.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return out.absolutePath
    }
}
