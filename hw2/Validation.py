def validation(k=1, l=None, u=None, eps1=None, eps2=None):
    valid = True
    error_message = ''
    if l is not None and u is not None:
        try:
            if l >= u:
                valid = False
                error_message = "The lower limit is bigger then the upper limit"
        except Exception as err:
            valid = False
            error_message += f' {str(err)},'
    if eps1 or eps2 is not None:
        try:
            if eps1 < 0 or k <= 0:
                valid = False
                error_message = "The difference and the iterations max number must be a positive number"
        except Exception as err:
            valid = False
            error_message += f' {str(err)},'
    if error_message != '':
        print(error_message)
    return valid
