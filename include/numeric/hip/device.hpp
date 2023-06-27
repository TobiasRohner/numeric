#ifndef NUMERIC_HIP_DEVICE_HPP_
#define NUMERIC_HIP_DEVICE_HPP_


namespace numeric::hip {

class Device {
public:
  Device();
  Device(int id);
  Device(const Device &) = default;
  Device &operator=(const Device &) = default;

  static int count();

  void activate() const;
  void sync() const;

  int id() const;

  template<typename Func>
  void do_while_active(const Func &func) const {
    Device current;
    activate();
    func();
    current.activate();
  }

private:
  int id_;
};

}


#endif
