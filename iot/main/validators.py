from django.core.exceptions import ValidationError

def file_size(value):
    filesize = value.size
    print(f"1file size is : {filesize}")
    if filesize > 419430400000: # 50MB 보다 크다면 예외 발생000
        print(f"2file size is : {filesize}")
        raise ValidationError("maximum size is 50mb")