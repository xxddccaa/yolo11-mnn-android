package com.yolomnn

import android.graphics.RectF

/** A single detection result in original-image pixel coordinates. */
data class Detection(
    val box: RectF,
    val classId: Int,
    val label: String,
    val score: Float,
)
