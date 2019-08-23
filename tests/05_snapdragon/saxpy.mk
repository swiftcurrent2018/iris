BRISBANE=$(HOME)/work/brisbane-rts

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := libOpenCL
LOCAL_SRC_FILES := ../libOpenCL.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libbrisbane
LOCAL_SRC_FILES:= ../obj/local/armeabi-v7a/libbrisbane.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE:=saxpy
LOCAL_STATIC_LIBRARIES := libbrisbane
LOCAL_SHARED_LIBRARIES := libOpenCL
LOCAL_C_INCLUDES:=$(BRISBANE)/include
LOCAL_SRC_FILES:= $(BRISBANE)/apps/saxpy/saxpy-brisbane.cpp
LOCAL_LDFLAGS:=-fopenmp
include $(BUILD_EXECUTABLE)

