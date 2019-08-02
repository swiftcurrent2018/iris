#ifndef BRISBANE_RT_SRC_MESSAGE_H
#define BRISBANE_RT_SRC_MESSAGE_H

#include <stdint.h>
#include <stdlib.h>

namespace brisbane {
namespace rt {

#define BRISBANE_MSG_SIZE           0x100

class Message {
public:
    Message(long header = -1);
    ~Message();

    void WriteHeader(int32_t v);
    void WriteBool(bool v);
    void WriteInt(int32_t v);
    void WriteUInt(uint32_t v);
    void WriteLong(int64_t v);
    void WriteULong(uint64_t v);
    void WriteFloat(float v);
    void WriteDouble(double v);
    void WriteString(const char* v);
    void WritePtr(void *ptr);
    void Write(const void* v, size_t size);

    int32_t ReadHeader();
    bool ReadBool();
    int32_t ReadInt();
    uint32_t ReadUInt();
    int64_t ReadLong();
    uint64_t ReadULong();
    float ReadFloat();
    double ReadDouble();
    char* ReadString();
    char* ReadString(size_t len);
    void* ReadPtr();
    void* Read(size_t size);

    char* buf() { return buf_; }
    void Clear();

private:
    char buf_[BRISBANE_MSG_SIZE] __attribute__ ((aligned(0x10)));
    size_t offset_;
};

} /* namespace rt */
} /* namespace brisbane */


#endif /*BRISBANE_RT_SRC_MESSAGE_H */
