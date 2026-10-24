// Auto-generated. Do not edit.
// Distilled decision tree (depth 12, 1024 leaves) approximating the offline
// GBM cost model for arch: xe3lpg.
//
// Generated from sweep timing data by training a gradient boosting cost model,
// labeling a dense problem grid, and distilling into a depth-capped decision tree.
//
// Arch-specific: dispatch on arch. One .inc per arch.
//
// Note: Common heuristic rules could potentially be derived from cross-arch analysis
// of raw sweep timing data across all Intel GPU architectures, which may reveal
// universal performance patterns applicable as a simpler baseline or fallback.
//
// Performance oracle only, like HeuristicTiling in
// subgroup_matrix_tiling_selector.cc: the caller MUST still run IsTilingValid.
// The tree does not see K's block count, so it can propose
// split_k > K / 16.
//
// The including TU must provide <cmath>, <cstdint> and SubgroupMatrixTiling.

namespace onnxruntime::webgpu::intel::xe3lpg {

inline SubgroupMatrixTiling PredictSgmmTilingTree(uint32_t M, uint32_t N, uint32_t K, uint32_t batch) {
  [[maybe_unused]] const float f0 = std::log2(static_cast<float>(M < 1u ? 1u : M));
  [[maybe_unused]] const float f1 = std::log2(static_cast<float>(N < 1u ? 1u : N));
  [[maybe_unused]] const float f2 = std::log2(static_cast<float>(K < 1u ? 1u : K));
  [[maybe_unused]] const float f3 = std::log2(static_cast<float>(batch < 1u ? 1u : batch));
  if (f1 <= 8.292481f) {
    if (f2 <= 7.292481f) {
      if (f2 <= 5.292481f) {
        if (f2 <= 4.500000f) {
          if (f0 <= 8.292481f) {
            if (f1 <= 4.292481f) {
              if (f3 <= 6.500000f) {
                return {8, 16, 1};
              } else {
                if (f0 <= 4.792481f) {
                  return {8, 16, 1};
                } else {
                  return {8, 32, 1};
                }
              }
            } else {
              if (f3 <= 4.500000f) {
                return {8, 32, 1};
              } else {
                if (f1 <= 5.292481f) {
                  return {8, 32, 1};
                } else {
                  if (f0 <= 6.292481f) {
                    return {8, 32, 1};
                  } else {
                    if (f1 <= 6.792481f) {
                      return {8, 32, 1};
                    } else {
                      return {16, 64, 1};
                    }
                  }
                }
              }
            }
          } else {
            if (f1 <= 5.292481f) {
              if (f3 <= 5.500000f) {
                return {8, 32, 1};
              } else {
                if (f1 <= 4.292481f) {
                  return {16, 16, 1};
                } else {
                  return {32, 32, 1};
                }
              }
            } else {
              if (f3 <= 2.500000f) {
                if (f0 <= 10.292481f) {
                  return {8, 32, 1};
                } else {
                  return {8, 64, 1};
                }
              } else {
                return {16, 64, 1};
              }
            }
          }
        } else {
          if (f3 <= 4.500000f) {
            if (f0 <= 7.292481f) {
              if (f1 <= 4.292481f) {
                return {8, 16, 2};
              } else {
                if (f0 <= 4.292481f) {
                  return {8, 16, 2};
                } else {
                  return {8, 32, 2};
                }
              }
            } else {
              if (f3 <= 1.500000f) {
                if (f1 <= 4.292481f) {
                  return {8, 16, 2};
                } else {
                  if (f0 <= 10.292481f) {
                    if (f1 <= 7.292481f) {
                      return {8, 32, 2};
                    } else {
                      return {8, 32, 1};
                    }
                  } else {
                    return {8, 64, 1};
                  }
                }
              } else {
                if (f1 <= 5.292481f) {
                  return {8, 32, 1};
                } else {
                  if (f0 <= 8.792481f) {
                    return {8, 32, 1};
                  } else {
                    if (f1 <= 6.292481f) {
                      return {8, 64, 1};
                    } else {
                      return {16, 64, 1};
                    }
                  }
                }
              }
            }
          } else {
            if (f1 <= 5.292481f) {
              if (f1 <= 4.292481f) {
                return {8, 16, 2};
              } else {
                if (f0 <= 10.792481f) {
                  return {8, 32, 1};
                } else {
                  return {16, 32, 1};
                }
              }
            } else {
              if (f0 <= 6.292481f) {
                return {8, 32, 1};
              } else {
                if (f1 <= 7.292481f) {
                  if (f0 <= 9.292481f) {
                    if (f3 <= 8.500000f) {
                      return {8, 32, 1};
                    } else {
                      return {16, 64, 1};
                    }
                  } else {
                    if (f3 <= 8.500000f) {
                      return {16, 64, 1};
                    } else {
                      return {32, 32, 2};
                    }
                  }
                } else {
                  if (f0 <= 10.292481f) {
                    return {16, 64, 1};
                  } else {
                    if (f3 <= 8.500000f) {
                      return {32, 64, 1};
                    } else {
                      return {8, 16, 1};
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if (f1 <= 5.292481f) {
          if (f1 <= 4.292481f) {
            if (f2 <= 5.792481f) {
              if (f3 <= 4.500000f) {
                return {8, 16, 2};
              } else {
                return {8, 16, 2};
              }
            } else {
              if (f3 <= 4.500000f) {
                if (f2 <= 6.792481f) {
                  return {8, 16, 4};
                } else {
                  if (f0 <= 7.792481f) {
                    return {8, 16, 8};
                  } else {
                    return {8, 16, 4};
                  }
                }
              } else {
                if (f2 <= 6.792481f) {
                  return {8, 16, 4};
                } else {
                  if (f0 <= 5.792481f) {
                    return {8, 16, 8};
                  } else {
                    return {8, 16, 2};
                  }
                }
              }
            }
          } else {
            if (f2 <= 6.292481f) {
              if (f3 <= 4.500000f) {
                if (f0 <= 8.292481f) {
                  if (f2 <= 5.792481f) {
                    if (f3 <= 2.500000f) {
                      return {8, 16, 2};
                    } else {
                      if (f0 <= 5.792481f) {
                        return {8, 16, 2};
                      } else {
                        return {8, 32, 2};
                      }
                    }
                  } else {
                    if (f3 <= 2.500000f) {
                      return {8, 16, 4};
                    } else {
                      if (f0 <= 5.292481f) {
                        return {8, 32, 4};
                      } else {
                        return {8, 32, 2};
                      }
                    }
                  }
                } else {
                  if (f3 <= 1.500000f) {
                    return {8, 32, 2};
                  } else {
                    return {8, 32, 1};
                  }
                }
              } else {
                if (f0 <= 8.792481f) {
                  if (f0 <= 6.792481f) {
                    return {8, 32, 1};
                  } else {
                    if (f3 <= 9.500000f) {
                      return {8, 32, 1};
                    } else {
                      return {16, 32, 2};
                    }
                  }
                } else {
                  if (f3 <= 7.500000f) {
                    if (f0 <= 11.292481f) {
                      return {8, 32, 1};
                    } else {
                      return {16, 32, 1};
                    }
                  } else {
                    return {16, 32, 1};
                  }
                }
              }
            } else {
              if (f3 <= 6.500000f) {
                if (f0 <= 6.792481f) {
                  if (f3 <= 2.500000f) {
                    if (f2 <= 6.792481f) {
                      return {8, 16, 4};
                    } else {
                      return {8, 16, 8};
                    }
                  } else {
                    if (f0 <= 4.792481f) {
                      return {8, 32, 4};
                    } else {
                      if (f3 <= 4.500000f) {
                        return {8, 32, 4};
                      } else {
                        return {16, 32, 4};
                      }
                    }
                  }
                } else {
                  if (f3 <= 4.500000f) {
                    return {8, 32, 4};
                  } else {
                    if (f0 <= 9.792481f) {
                      return {8, 32, 4};
                    } else {
                      return {16, 32, 2};
                    }
                  }
                }
              } else {
                if (f0 <= 5.792481f) {
                  return {8, 32, 4};
                } else {
                  if (f0 <= 10.792481f) {
                    return {8, 32, 1};
                  } else {
                    if (f3 <= 8.500000f) {
                      return {16, 32, 2};
                    } else {
                      return {32, 32, 2};
                    }
                  }
                }
              }
            }
          }
        } else {
          if (f3 <= 4.500000f) {
            if (f2 <= 6.292481f) {
              if (f0 <= 10.292481f) {
                if (f0 <= 7.292481f) {
                  if (f2 <= 5.792481f) {
                    if (f0 <= 5.292481f) {
                      if (f3 <= 2.500000f) {
                        return {8, 16, 2};
                      } else {
                        return {8, 32, 2};
                      }
                    } else {
                      return {8, 32, 2};
                    }
                  } else {
                    if (f3 <= 2.500000f) {
                      if (f0 <= 6.292481f) {
                        return {8, 16, 4};
                      } else {
                        return {8, 32, 4};
                      }
                    } else {
                      if (f0 <= 4.792481f) {
                        return {8, 32, 4};
                      } else {
                        return {8, 32, 2};
                      }
                    }
                  }
                } else {
                  if (f1 <= 7.292481f) {
                    if (f2 <= 5.792481f) {
                      if (f3 <= 1.500000f) {
                        return {8, 32, 2};
                      } else {
                        return {16, 32, 1};
                      }
                    } else {
                      if (f3 <= 2.500000f) {
                        return {16, 32, 4};
                      } else {
                        return {16, 32, 1};
                      }
                    }
                  } else {
                    if (f3 <= 2.500000f) {
                      return {16, 32, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              } else {
                if (f1 <= 7.292481f) {
                  return {8, 64, 1};
                } else {
                  if (f3 <= 0.500000f) {
                    return {8, 64, 1};
                  } else {
                    return {16, 64, 1};
                  }
                }
              }
            } else {
              if (f0 <= 10.292481f) {
                if (f0 <= 5.792481f) {
                  if (f3 <= 1.500000f) {
                    if (f2 <= 6.792481f) {
                      return {8, 16, 4};
                    } else {
                      return {8, 16, 8};
                    }
                  } else {
                    if (f1 <= 5.792481f) {
                      return {16, 16, 4};
                    } else {
                      if (f3 <= 3.500000f) {
                        return {8, 32, 4};
                      } else {
                        if (f0 <= 3.292481f) {
                          return {8, 32, 4};
                        } else {
                          return {16, 32, 4};
                        }
                      }
                    }
                  }
                } else {
                  if (f1 <= 5.792481f) {
                    return {16, 32, 4};
                  } else {
                    if (f1 <= 7.292481f) {
                      if (f3 <= 1.500000f) {
                        if (f0 <= 7.792481f) {
                          return {8, 32, 4};
                        } else {
                          return {16, 32, 4};
                        }
                      } else {
                        return {16, 32, 4};
                      }
                    } else {
                      if (f3 <= 1.500000f) {
                        return {16, 32, 4};
                      } else {
                        if (f0 <= 8.292481f) {
                          return {16, 32, 4};
                        } else {
                          return {16, 64, 1};
                        }
                      }
                    }
                  }
                }
              } else {
                if (f1 <= 6.792481f) {
                  if (f3 <= 2.500000f) {
                    return {8, 64, 4};
                  } else {
                    return {16, 64, 1};
                  }
                } else {
                  return {16, 64, 1};
                }
              }
            }
          } else {
            if (f0 <= 6.292481f) {
              if (f2 <= 6.292481f) {
                if (f0 <= 5.292481f) {
                  if (f1 <= 6.792481f) {
                    if (f1 <= 5.792481f) {
                      return {16, 16, 1};
                    } else {
                      return {8, 32, 1};
                    }
                  } else {
                    if (f3 <= 7.500000f) {
                      if (f0 <= 4.292481f) {
                        if (f0 <= 3.292481f) {
                          return {8, 32, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        return {32, 32, 2};
                      }
                    } else {
                      return {16, 32, 2};
                    }
                  }
                } else {
                  if (f3 <= 6.500000f) {
                    return {16, 32, 1};
                  } else {
                    if (f1 <= 7.292481f) {
                      return {8, 64, 1};
                    } else {
                      return {16, 64, 1};
                    }
                  }
                }
              } else {
                if (f3 <= 6.500000f) {
                  if (f0 <= 4.792481f) {
                    return {16, 32, 4};
                  } else {
                    return {16, 32, 4};
                  }
                } else {
                  if (f1 <= 6.792481f) {
                    if (f0 <= 4.292481f) {
                      if (f2 <= 6.792481f) {
                        return {8, 16, 4};
                      } else {
                        return {8, 16, 8};
                      }
                    } else {
                      return {8, 64, 4};
                    }
                  } else {
                    if (f0 <= 5.292481f) {
                      return {16, 64, 1};
                    } else {
                      return {16, 64, 1};
                    }
                  }
                }
              }
            } else {
              if (f1 <= 7.292481f) {
                if (f0 <= 10.792481f) {
                  if (f3 <= 6.500000f) {
                    if (f0 <= 8.792481f) {
                      if (f2 <= 6.292481f) {
                        return {16, 32, 1};
                      } else {
                        return {16, 32, 4};
                      }
                    } else {
                      return {16, 64, 1};
                    }
                  } else {
                    if (f1 <= 5.792481f) {
                      if (f0 <= 6.792481f) {
                        return {8, 64, 1};
                      } else {
                        return {16, 64, 1};
                      }
                    } else {
                      return {16, 64, 1};
                    }
                  }
                } else {
                  if (f3 <= 8.500000f) {
                    return {16, 64, 1};
                  } else {
                    return {32, 64, 1};
                  }
                }
              } else {
                if (f0 <= 8.792481f) {
                  return {16, 64, 1};
                } else {
                  if (f2 <= 6.292481f) {
                    if (f3 <= 7.500000f) {
                      return {16, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    return {32, 64, 1};
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if (f3 <= 2.500000f) {
        if (f0 <= 7.292481f) {
          if (f1 <= 5.292481f) {
            if (f0 <= 4.792481f) {
              if (f0 <= 3.292481f) {
                if (f3 <= 1.500000f) {
                  if (f2 <= 9.292481f) {
                    return {8, 16, 8};
                  } else {
                    if (f2 <= 15.292481f) {
                      return {16, 16, 8};
                    } else {
                      return {8, 16, 8};
                    }
                  }
                } else {
                  return {8, 16, 8};
                }
              } else {
                return {8, 16, 8};
              }
            } else {
              if (f2 <= 15.292481f) {
                if (f3 <= 0.500000f) {
                  if (f0 <= 5.792481f) {
                    return {8, 16, 8};
                  } else {
                    if (f1 <= 4.292481f) {
                      if (f0 <= 6.792481f) {
                        return {8, 16, 8};
                      } else {
                        return {16, 16, 8};
                      }
                    } else {
                      if (f2 <= 8.292481f) {
                        return {8, 16, 8};
                      } else {
                        if (f0 <= 6.792481f) {
                          return {16, 16, 8};
                        } else {
                          return {8, 16, 8};
                        }
                      }
                    }
                  }
                } else {
                  if (f2 <= 9.292481f) {
                    return {16, 16, 8};
                  } else {
                    return {16, 16, 8};
                  }
                }
              } else {
                return {8, 16, 8};
              }
            }
          } else {
            if (f2 <= 15.292481f) {
              if (f0 <= 4.792481f) {
                if (f1 <= 7.792481f) {
                  if (f1 <= 6.292481f) {
                    if (f3 <= 0.500000f) {
                      if (f0 <= 3.292481f) {
                        return {16, 16, 8};
                      } else {
                        return {8, 16, 8};
                      }
                    } else {
                      return {16, 16, 8};
                    }
                  } else {
                    if (f3 <= 1.500000f) {
                      return {16, 16, 8};
                    } else {
                      if (f0 <= 3.292481f) {
                        return {8, 16, 8};
                      } else {
                        if (f2 <= 10.292481f) {
                          return {16, 16, 8};
                        } else {
                          return {16, 16, 8};
                        }
                      }
                    }
                  }
                } else {
                  if (f2 <= 10.292481f) {
                    return {8, 32, 8};
                  } else {
                    if (f0 <= 4.292481f) {
                      if (f0 <= 3.292481f) {
                        return {8, 16, 8};
                      } else {
                        return {16, 16, 8};
                      }
                    } else {
                      return {8, 16, 8};
                    }
                  }
                }
              } else {
                if (f3 <= 1.500000f) {
                  if (f1 <= 7.292481f) {
                    if (f2 <= 9.292481f) {
                      if (f2 <= 8.292481f) {
                        return {16, 16, 8};
                      } else {
                        return {16, 16, 8};
                      }
                    } else {
                      return {16, 16, 8};
                    }
                  } else {
                    if (f2 <= 10.292481f) {
                      return {16, 32, 8};
                    } else {
                      if (f0 <= 6.292481f) {
                        return {16, 16, 8};
                      } else {
                        if (f3 <= 0.500000f) {
                          return {16, 16, 8};
                        } else {
                          return {16, 32, 8};
                        }
                      }
                    }
                  }
                } else {
                  if (f1 <= 6.792481f) {
                    if (f2 <= 10.292481f) {
                      if (f2 <= 8.292481f) {
                        return {16, 32, 4};
                      } else {
                        return {16, 64, 8};
                      }
                    } else {
                      if (f0 <= 6.292481f) {
                        return {16, 16, 8};
                      } else {
                        if (f1 <= 6.292481f) {
                          return {16, 16, 8};
                        } else {
                          return {16, 32, 8};
                        }
                      }
                    }
                  } else {
                    if (f2 <= 8.292481f) {
                      return {16, 32, 4};
                    } else {
                      if (f2 <= 13.292481f) {
                        if (f2 <= 10.292481f) {
                          return {16, 64, 8};
                        } else {
                          return {32, 16, 8};
                        }
                      } else {
                        return {16, 64, 8};
                      }
                    }
                  }
                }
              }
            } else {
              if (f0 <= 5.792481f) {
                return {8, 16, 8};
              } else {
                if (f1 <= 6.792481f) {
                  return {8, 16, 8};
                } else {
                  return {8, 64, 8};
                }
              }
            }
          }
        } else {
          if (f1 <= 5.292481f) {
            if (f1 <= 4.292481f) {
              if (f0 <= 10.292481f) {
                if (f2 <= 12.292481f) {
                  return {8, 16, 8};
                } else {
                  return {16, 16, 8};
                }
              } else {
                if (f2 <= 14.292481f) {
                  if (f2 <= 8.292481f) {
                    return {16, 32, 4};
                  } else {
                    return {8, 16, 8};
                  }
                } else {
                  return {64, 32, 4};
                }
              }
            } else {
              if (f0 <= 8.792481f) {
                if (f2 <= 8.292481f) {
                  return {8, 32, 8};
                } else {
                  if (f2 <= 12.292481f) {
                    if (f2 <= 9.292481f) {
                      return {16, 32, 8};
                    } else {
                      return {8, 16, 8};
                    }
                  } else {
                    if (f3 <= 1.500000f) {
                      if (f2 <= 15.292481f) {
                        return {16, 16, 8};
                      } else {
                        return {8, 16, 8};
                      }
                    } else {
                      if (f0 <= 7.792481f) {
                        return {16, 16, 8};
                      } else {
                        return {16, 32, 8};
                      }
                    }
                  }
                }
              } else {
                if (f2 <= 15.292481f) {
                  if (f3 <= 1.500000f) {
                    if (f0 <= 10.292481f) {
                      if (f2 <= 10.292481f) {
                        return {8, 32, 8};
                      } else {
                        if (f3 <= 0.500000f) {
                          return {8, 32, 8};
                        } else {
                          return {16, 32, 8};
                        }
                      }
                    } else {
                      if (f2 <= 8.292481f) {
                        return {8, 32, 8};
                      } else {
                        return {8, 32, 8};
                      }
                    }
                  } else {
                    if (f2 <= 11.292481f) {
                      if (f0 <= 9.792481f) {
                        if (f2 <= 10.292481f) {
                          return {8, 32, 8};
                        } else {
                          return {16, 32, 8};
                        }
                      } else {
                        return {16, 32, 8};
                      }
                    } else {
                      if (f0 <= 10.292481f) {
                        return {32, 32, 8};
                      } else {
                        return {8, 32, 8};
                      }
                    }
                  }
                } else {
                  return {32, 32, 8};
                }
              }
            }
          } else {
            if (f2 <= 11.292481f) {
              if (f0 <= 9.792481f) {
                if (f3 <= 1.500000f) {
                  if (f1 <= 5.792481f) {
                    if (f0 <= 9.292481f) {
                      return {16, 16, 8};
                    } else {
                      return {16, 32, 8};
                    }
                  } else {
                    return {16, 32, 8};
                  }
                } else {
                  if (f2 <= 8.292481f) {
                    return {16, 32, 4};
                  } else {
                    if (f1 <= 6.292481f) {
                      return {16, 32, 8};
                    } else {
                      if (f0 <= 8.292481f) {
                        if (f2 <= 10.292481f) {
                          return {16, 32, 8};
                        } else {
                          return {32, 16, 8};
                        }
                      } else {
                        return {32, 32, 4};
                      }
                    }
                  }
                }
              } else {
                if (f2 <= 8.292481f) {
                  return {16, 32, 4};
                } else {
                  if (f3 <= 1.500000f) {
                    if (f0 <= 11.292481f) {
                      if (f1 <= 5.792481f) {
                        return {16, 16, 8};
                      } else {
                        if (f1 <= 6.792481f) {
                          return {16, 32, 8};
                        } else {
                          return {32, 32, 4};
                        }
                      }
                    } else {
                      return {16, 64, 4};
                    }
                  } else {
                    if (f0 <= 11.292481f) {
                      if (f1 <= 5.792481f) {
                        return {32, 16, 8};
                      } else {
                        return {32, 32, 4};
                      }
                    } else {
                      return {32, 32, 4};
                    }
                  }
                }
              }
            } else {
              if (f0 <= 10.292481f) {
                if (f2 <= 13.292481f) {
                  if (f3 <= 1.500000f) {
                    if (f1 <= 6.292481f) {
                      if (f0 <= 8.292481f) {
                        return {16, 16, 8};
                      } else {
                        return {16, 32, 8};
                      }
                    } else {
                      return {32, 32, 8};
                    }
                  } else {
                    return {32, 64, 8};
                  }
                } else {
                  if (f2 <= 15.292481f) {
                    if (f3 <= 0.500000f) {
                      if (f0 <= 8.792481f) {
                        if (f1 <= 6.792481f) {
                          return {16, 16, 8};
                        } else {
                          return {16, 32, 8};
                        }
                      } else {
                        if (f2 <= 14.292481f) {
                          return {16, 32, 8};
                        } else {
                          return {16, 64, 8};
                        }
                      }
                    } else {
                      if (f0 <= 8.792481f) {
                        return {16, 64, 8};
                      } else {
                        if (f1 <= 6.292481f) {
                          return {16, 64, 8};
                        } else {
                          return {64, 32, 4};
                        }
                      }
                    }
                  } else {
                    return {16, 64, 8};
                  }
                }
              } else {
                if (f2 <= 15.292481f) {
                  if (f2 <= 14.292481f) {
                    if (f1 <= 7.292481f) {
                      if (f2 <= 13.292481f) {
                        if (f0 <= 11.292481f) {
                          return {32, 64, 8};
                        } else {
                          return {8, 64, 8};
                        }
                      } else {
                        return {8, 64, 8};
                      }
                    } else {
                      if (f3 <= 1.500000f) {
                        return {16, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  } else {
                    return {8, 64, 8};
                  }
                } else {
                  if (f1 <= 6.792481f) {
                    return {32, 64, 8};
                  } else {
                    return {8, 64, 4};
                  }
                }
              }
            }
          }
        }
      } else {
        if (f1 <= 6.792481f) {
          if (f0 <= 5.792481f) {
            if (f3 <= 5.500000f) {
              if (f0 <= 3.292481f) {
                if (f1 <= 5.792481f) {
                  if (f3 <= 4.500000f) {
                    return {8, 16, 8};
                  } else {
                    return {8, 16, 8};
                  }
                } else {
                  if (f3 <= 3.500000f) {
                    return {8, 16, 8};
                  } else {
                    return {8, 32, 8};
                  }
                }
              } else {
                if (f2 <= 8.292481f) {
                  if (f1 <= 4.292481f) {
                    return {8, 16, 8};
                  } else {
                    return {8, 32, 8};
                  }
                } else {
                  if (f1 <= 5.292481f) {
                    if (f0 <= 4.292481f) {
                      return {16, 16, 8};
                    } else {
                      if (f0 <= 4.792481f) {
                        if (f3 <= 3.500000f) {
                          return {8, 16, 8};
                        } else {
                          return {8, 16, 8};
                        }
                      } else {
                        if (f2 <= 11.292481f) {
                          return {8, 16, 8};
                        } else {
                          return {16, 16, 8};
                        }
                      }
                    }
                  } else {
                    if (f1 <= 5.792481f) {
                      if (f0 <= 5.292481f) {
                        if (f3 <= 4.500000f) {
                          return {16, 16, 8};
                        } else {
                          return {16, 16, 8};
                        }
                      } else {
                        return {16, 64, 8};
                      }
                    } else {
                      if (f3 <= 3.500000f) {
                        if (f2 <= 15.292481f) {
                          return {16, 16, 8};
                        } else {
                          return {8, 16, 8};
                        }
                      } else {
                        if (f2 <= 11.292481f) {
                          return {8, 16, 8};
                        } else {
                          return {16, 32, 8};
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if (f1 <= 5.292481f) {
                if (f1 <= 4.292481f) {
                  if (f2 <= 11.292481f) {
                    if (f3 <= 8.500000f) {
                      return {8, 16, 8};
                    } else {
                      if (f2 <= 9.292481f) {
                        return {8, 16, 8};
                      } else {
                        return {8, 16, 2};
                      }
                    }
                  } else {
                    if (f0 <= 3.292481f) {
                      return {8, 16, 4};
                    } else {
                      if (f0 <= 4.292481f) {
                        return {16, 16, 8};
                      } else {
                        return {8, 16, 8};
                      }
                    }
                  }
                } else {
                  if (f2 <= 11.292481f) {
                    if (f3 <= 7.500000f) {
                      if (f0 <= 4.792481f) {
                        return {8, 16, 8};
                      } else {
                        return {8, 32, 8};
                      }
                    } else {
                      if (f2 <= 8.292481f) {
                        if (f3 <= 8.500000f) {
                          return {8, 16, 8};
                        } else {
                          return {8, 32, 8};
                        }
                      } else {
                        if (f0 <= 4.792481f) {
                          return {8, 16, 2};
                        } else {
                          return {32, 32, 4};
                        }
                      }
                    }
                  } else {
                    if (f2 <= 15.292481f) {
                      if (f0 <= 3.292481f) {
                        if (f3 <= 9.500000f) {
                          return {8, 32, 4};
                        } else {
                          return {8, 32, 8};
                        }
                      } else {
                        if (f0 <= 4.792481f) {
                          return {8, 32, 8};
                        } else {
                          return {8, 32, 8};
                        }
                      }
                    } else {
                      return {16, 32, 8};
                    }
                  }
                }
              } else {
                if (f0 <= 4.292481f) {
                  if (f2 <= 11.292481f) {
                    if (f0 <= 3.292481f) {
                      return {8, 16, 8};
                    } else {
                      if (f3 <= 6.500000f) {
                        return {16, 16, 8};
                      } else {
                        return {16, 16, 1};
                      }
                    }
                  } else {
                    if (f1 <= 6.292481f) {
                      if (f1 <= 5.792481f) {
                        if (f2 <= 12.292481f) {
                          return {8, 16, 2};
                        } else {
                          return {8, 16, 4};
                        }
                      } else {
                        if (f3 <= 6.500000f) {
                          return {8, 64, 8};
                        } else {
                          return {8, 64, 4};
                        }
                      }
                    } else {
                      if (f0 <= 3.292481f) {
                        return {8, 32, 2};
                      } else {
                        if (f2 <= 15.292481f) {
                          return {16, 32, 4};
                        } else {
                          return {16, 32, 2};
                        }
                      }
                    }
                  }
                } else {
                  if (f0 <= 4.792481f) {
                    if (f2 <= 15.292481f) {
                      if (f2 <= 11.292481f) {
                        if (f3 <= 8.500000f) {
                          return {32, 16, 8};
                        } else {
                          return {8, 64, 4};
                        }
                      } else {
                        return {8, 64, 4};
                      }
                    } else {
                      return {8, 64, 2};
                    }
                  } else {
                    if (f0 <= 5.292481f) {
                      if (f1 <= 5.792481f) {
                        if (f3 <= 6.500000f) {
                          return {32, 16, 8};
                        } else {
                          return {32, 16, 4};
                        }
                      } else {
                        if (f1 <= 6.292481f) {
                          return {8, 64, 4};
                        } else {
                          return {32, 32, 4};
                        }
                      }
                    } else {
                      if (f2 <= 12.292481f) {
                        return {16, 64, 4};
                      } else {
                        if (f2 <= 15.292481f) {
                          return {16, 64, 8};
                        } else {
                          return {16, 64, 4};
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if (f2 <= 11.292481f) {
              if (f1 <= 4.292481f) {
                if (f3 <= 6.500000f) {
                  if (f0 <= 9.792481f) {
                    return {8, 16, 8};
                  } else {
                    if (f2 <= 9.292481f) {
                      if (f3 <= 4.500000f) {
                        return {8, 16, 8};
                      } else {
                        return {32, 16, 2};
                      }
                    } else {
                      return {64, 16, 8};
                    }
                  }
                } else {
                  if (f0 <= 6.792481f) {
                    return {8, 16, 8};
                  } else {
                    if (f3 <= 8.500000f) {
                      if (f0 <= 9.792481f) {
                        if (f2 <= 8.292481f) {
                          return {64, 16, 8};
                        } else {
                          return {8, 16, 4};
                        }
                      } else {
                        return {64, 32, 8};
                      }
                    } else {
                      return {64, 16, 8};
                    }
                  }
                }
              } else {
                if (f2 <= 8.292481f) {
                  if (f3 <= 6.500000f) {
                    if (f0 <= 10.792481f) {
                      if (f1 <= 5.292481f) {
                        return {8, 32, 8};
                      } else {
                        if (f1 <= 5.792481f) {
                          return {32, 16, 4};
                        } else {
                          return {16, 32, 4};
                        }
                      }
                    } else {
                      return {32, 32, 2};
                    }
                  } else {
                    if (f1 <= 5.292481f) {
                      if (f0 <= 8.792481f) {
                        return {8, 32, 1};
                      } else {
                        return {32, 32, 2};
                      }
                    } else {
                      if (f0 <= 10.792481f) {
                        return {16, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  }
                } else {
                  if (f3 <= 6.500000f) {
                    if (f0 <= 8.292481f) {
                      if (f3 <= 4.500000f) {
                        if (f1 <= 5.292481f) {
                          return {8, 16, 8};
                        } else {
                          return {32, 16, 8};
                        }
                      } else {
                        if (f0 <= 6.792481f) {
                          return {32, 16, 8};
                        } else {
                          return {32, 32, 4};
                        }
                      }
                    } else {
                      if (f0 <= 10.292481f) {
                        if (f1 <= 5.792481f) {
                          return {32, 32, 4};
                        } else {
                          return {32, 32, 4};
                        }
                      } else {
                        if (f1 <= 5.292481f) {
                          return {32, 32, 4};
                        } else {
                          return {64, 32, 4};
                        }
                      }
                    }
                  } else {
                    if (f1 <= 5.292481f) {
                      if (f0 <= 7.292481f) {
                        if (f3 <= 8.500000f) {
                          return {32, 32, 4};
                        } else {
                          return {8, 32, 1};
                        }
                      } else {
                        if (f0 <= 8.792481f) {
                          return {8, 32, 8};
                        } else {
                          return {64, 32, 8};
                        }
                      }
                    } else {
                      if (f2 <= 9.292481f) {
                        if (f1 <= 6.292481f) {
                          return {16, 64, 2};
                        } else {
                          return {32, 32, 2};
                        }
                      } else {
                        if (f0 <= 9.292481f) {
                          return {64, 32, 4};
                        } else {
                          return {64, 32, 4};
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if (f1 <= 5.292481f) {
                if (f2 <= 13.292481f) {
                  if (f1 <= 4.292481f) {
                    if (f0 <= 8.292481f) {
                      return {16, 16, 8};
                    } else {
                      if (f3 <= 5.500000f) {
                        if (f0 <= 10.792481f) {
                          return {8, 16, 8};
                        } else {
                          return {64, 16, 8};
                        }
                      } else {
                        if (f2 <= 12.292481f) {
                          return {64, 32, 8};
                        } else {
                          return {8, 16, 4};
                        }
                      }
                    }
                  } else {
                    if (f3 <= 5.500000f) {
                      if (f0 <= 11.292481f) {
                        if (f0 <= 6.792481f) {
                          return {16, 16, 8};
                        } else {
                          return {32, 32, 8};
                        }
                      } else {
                        if (f3 <= 3.500000f) {
                          return {64, 32, 2};
                        } else {
                          return {64, 32, 8};
                        }
                      }
                    } else {
                      if (f3 <= 7.500000f) {
                        if (f0 <= 8.792481f) {
                          return {8, 32, 8};
                        } else {
                          return {64, 32, 2};
                        }
                      } else {
                        if (f0 <= 7.292481f) {
                          return {32, 32, 4};
                        } else {
                          return {64, 32, 8};
                        }
                      }
                    }
                  }
                } else {
                  if (f2 <= 15.292481f) {
                    if (f0 <= 9.292481f) {
                      if (f3 <= 6.500000f) {
                        if (f1 <= 4.292481f) {
                          return {64, 16, 8};
                        } else {
                          return {32, 32, 8};
                        }
                      } else {
                        if (f0 <= 6.792481f) {
                          return {32, 32, 8};
                        } else {
                          return {64, 32, 4};
                        }
                      }
                    } else {
                      if (f3 <= 6.500000f) {
                        if (f3 <= 5.500000f) {
                          return {64, 32, 4};
                        } else {
                          return {64, 32, 2};
                        }
                      } else {
                        if (f0 <= 10.292481f) {
                          return {8, 16, 4};
                        } else {
                          return {8, 64, 4};
                        }
                      }
                    }
                  } else {
                    if (f0 <= 9.792481f) {
                      if (f3 <= 5.500000f) {
                        if (f1 <= 4.292481f) {
                          return {64, 16, 8};
                        } else {
                          return {32, 32, 8};
                        }
                      } else {
                        if (f3 <= 8.500000f) {
                          return {64, 32, 2};
                        } else {
                          return {8, 16, 2};
                        }
                      }
                    } else {
                      if (f3 <= 5.500000f) {
                        if (f3 <= 3.500000f) {
                          return {64, 32, 4};
                        } else {
                          return {64, 32, 2};
                        }
                      } else {
                        return {8, 64, 2};
                      }
                    }
                  }
                }
              } else {
                if (f2 <= 15.292481f) {
                  if (f0 <= 9.292481f) {
                    if (f3 <= 7.500000f) {
                      if (f1 <= 6.292481f) {
                        if (f1 <= 5.792481f) {
                          return {16, 64, 8};
                        } else {
                          return {16, 64, 8};
                        }
                      } else {
                        if (f3 <= 5.500000f) {
                          return {64, 32, 8};
                        } else {
                          return {64, 32, 2};
                        }
                      }
                    } else {
                      if (f1 <= 6.292481f) {
                        if (f2 <= 12.292481f) {
                          return {64, 32, 4};
                        } else {
                          return {32, 64, 8};
                        }
                      } else {
                        return {64, 32, 4};
                      }
                    }
                  } else {
                    if (f3 <= 6.500000f) {
                      if (f2 <= 14.292481f) {
                        if (f1 <= 6.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {64, 32, 2};
                        }
                      } else {
                        if (f0 <= 10.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {8, 64, 8};
                        }
                      }
                    } else {
                      if (f2 <= 12.292481f) {
                        if (f1 <= 6.292481f) {
                          return {32, 64, 8};
                        } else {
                          return {64, 32, 4};
                        }
                      } else {
                        if (f1 <= 6.292481f) {
                          return {8, 64, 4};
                        } else {
                          return {8, 64, 4};
                        }
                      }
                    }
                  }
                } else {
                  if (f0 <= 10.292481f) {
                    if (f3 <= 5.500000f) {
                      return {64, 32, 4};
                    } else {
                      if (f3 <= 9.500000f) {
                        if (f0 <= 8.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        if (f0 <= 8.292481f) {
                          return {8, 16, 1};
                        } else {
                          return {8, 64, 2};
                        }
                      }
                    }
                  } else {
                    if (f3 <= 5.500000f) {
                      if (f1 <= 6.292481f) {
                        return {32, 64, 1};
                      } else {
                        return {8, 64, 2};
                      }
                    } else {
                      return {8, 64, 2};
                    }
                  }
                }
              }
            }
          }
        } else {
          if (f0 <= 6.292481f) {
            if (f0 <= 3.292481f) {
              if (f3 <= 5.500000f) {
                return {8, 64, 8};
              } else {
                if (f2 <= 9.292481f) {
                  return {8, 64, 8};
                } else {
                  return {8, 64, 2};
                }
              }
            } else {
              if (f3 <= 5.500000f) {
                if (f2 <= 12.292481f) {
                  if (f2 <= 8.292481f) {
                    return {16, 32, 4};
                  } else {
                    if (f0 <= 4.292481f) {
                      return {16, 16, 8};
                    } else {
                      if (f3 <= 4.500000f) {
                        if (f2 <= 10.292481f) {
                          return {16, 16, 8};
                        } else {
                          return {32, 16, 8};
                        }
                      } else {
                        if (f2 <= 11.292481f) {
                          return {32, 32, 4};
                        } else {
                          return {32, 64, 8};
                        }
                      }
                    }
                  }
                } else {
                  if (f2 <= 13.292481f) {
                    return {16, 64, 8};
                  } else {
                    if (f2 <= 15.292481f) {
                      return {16, 64, 8};
                    } else {
                      return {16, 64, 8};
                    }
                  }
                }
              } else {
                if (f2 <= 11.292481f) {
                  if (f0 <= 4.792481f) {
                    if (f3 <= 8.500000f) {
                      if (f0 <= 4.292481f) {
                        return {16, 64, 4};
                      } else {
                        return {8, 64, 4};
                      }
                    } else {
                      if (f2 <= 8.292481f) {
                        return {16, 64, 8};
                      } else {
                        return {8, 64, 4};
                      }
                    }
                  } else {
                    if (f3 <= 7.500000f) {
                      if (f2 <= 8.292481f) {
                        return {16, 32, 4};
                      } else {
                        return {32, 32, 4};
                      }
                    } else {
                      if (f2 <= 9.292481f) {
                        if (f0 <= 5.292481f) {
                          return {16, 64, 2};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        if (f0 <= 5.292481f) {
                          return {8, 64, 4};
                        } else {
                          return {16, 64, 2};
                        }
                      }
                    }
                  }
                } else {
                  if (f0 <= 4.792481f) {
                    if (f3 <= 8.500000f) {
                      if (f2 <= 15.292481f) {
                        if (f2 <= 12.292481f) {
                          return {8, 64, 2};
                        } else {
                          return {8, 64, 4};
                        }
                      } else {
                        return {8, 64, 2};
                      }
                    } else {
                      if (f2 <= 15.292481f) {
                        if (f0 <= 4.292481f) {
                          return {16, 64, 8};
                        } else {
                          return {8, 64, 8};
                        }
                      } else {
                        return {8, 64, 2};
                      }
                    }
                  } else {
                    if (f2 <= 13.292481f) {
                      if (f0 <= 5.292481f) {
                        return {16, 64, 2};
                      } else {
                        if (f3 <= 7.500000f) {
                          return {8, 64, 8};
                        } else {
                          return {16, 64, 8};
                        }
                      }
                    } else {
                      if (f3 <= 6.500000f) {
                        return {16, 64, 2};
                      } else {
                        if (f3 <= 9.500000f) {
                          return {32, 64, 1};
                        } else {
                          return {8, 16, 1};
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if (f2 <= 11.292481f) {
              if (f0 <= 8.792481f) {
                if (f3 <= 6.500000f) {
                  if (f2 <= 8.292481f) {
                    return {16, 32, 4};
                  } else {
                    if (f3 <= 3.500000f) {
                      if (f0 <= 7.292481f) {
                        return {32, 16, 8};
                      } else {
                        return {32, 32, 4};
                      }
                    } else {
                      if (f3 <= 5.500000f) {
                        return {32, 32, 4};
                      } else {
                        return {32, 32, 4};
                      }
                    }
                  }
                } else {
                  if (f2 <= 9.292481f) {
                    return {16, 64, 1};
                  } else {
                    if (f3 <= 8.500000f) {
                      if (f2 <= 10.292481f) {
                        return {32, 32, 4};
                      } else {
                        return {16, 64, 2};
                      }
                    } else {
                      return {64, 32, 4};
                    }
                  }
                }
              } else {
                if (f3 <= 7.500000f) {
                  if (f2 <= 9.292481f) {
                    if (f2 <= 8.292481f) {
                      if (f1 <= 7.292481f) {
                        return {32, 32, 2};
                      } else {
                        if (f0 <= 10.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    } else {
                      return {32, 32, 2};
                    }
                  } else {
                    if (f0 <= 9.792481f) {
                      if (f3 <= 3.500000f) {
                        return {32, 32, 4};
                      } else {
                        if (f1 <= 7.292481f) {
                          return {32, 32, 4};
                        } else {
                          return {64, 32, 4};
                        }
                      }
                    } else {
                      if (f2 <= 10.292481f) {
                        return {64, 32, 4};
                      } else {
                        if (f3 <= 4.500000f) {
                          return {16, 64, 2};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  }
                } else {
                  if (f3 <= 9.500000f) {
                    if (f0 <= 9.792481f) {
                      if (f2 <= 9.292481f) {
                        return {16, 64, 1};
                      } else {
                        if (f1 <= 7.292481f) {
                          return {64, 32, 4};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    if (f0 <= 10.792481f) {
                      return {32, 64, 1};
                    } else {
                      if (f2 <= 9.292481f) {
                        return {8, 16, 2};
                      } else {
                        return {8, 16, 4};
                      }
                    }
                  }
                }
              }
            } else {
              if (f0 <= 10.292481f) {
                if (f1 <= 7.292481f) {
                  if (f2 <= 15.292481f) {
                    if (f0 <= 8.792481f) {
                      return {32, 64, 8};
                    } else {
                      if (f3 <= 4.500000f) {
                        if (f2 <= 13.292481f) {
                          return {16, 64, 8};
                        } else {
                          return {64, 32, 2};
                        }
                      } else {
                        if (f2 <= 12.292481f) {
                          return {64, 32, 4};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  } else {
                    if (f3 <= 4.500000f) {
                      return {32, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  }
                } else {
                  if (f3 <= 4.500000f) {
                    if (f0 <= 8.292481f) {
                      if (f2 <= 13.292481f) {
                        return {32, 64, 8};
                      } else {
                        return {64, 32, 4};
                      }
                    } else {
                      if (f2 <= 12.292481f) {
                        return {32, 32, 2};
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  } else {
                    if (f2 <= 14.292481f) {
                      if (f0 <= 7.292481f) {
                        if (f3 <= 7.500000f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              } else {
                if (f2 <= 13.292481f) {
                  if (f3 <= 9.500000f) {
                    if (f1 <= 7.292481f) {
                      return {32, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    if (f0 <= 11.292481f) {
                      return {32, 64, 1};
                    } else {
                      return {8, 64, 4};
                    }
                  }
                } else {
                  if (f2 <= 15.292481f) {
                    if (f2 <= 14.292481f) {
                      if (f3 <= 8.500000f) {
                        if (f3 <= 5.500000f) {
                          return {32, 64, 1};
                        } else {
                          return {8, 64, 2};
                        }
                      } else {
                        return {8, 64, 4};
                      }
                    } else {
                      return {8, 64, 8};
                    }
                  } else {
                    return {8, 64, 2};
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if (f0 <= 6.292481f) {
      if (f2 <= 12.292481f) {
        if (f2 <= 6.292481f) {
          if (f0 <= 5.292481f) {
            if (f2 <= 5.292481f) {
              if (f0 <= 3.292481f) {
                if (f3 <= 7.500000f) {
                  if (f2 <= 4.500000f) {
                    if (f3 <= 0.500000f) {
                      return {8, 32, 1};
                    } else {
                      return {16, 32, 1};
                    }
                  } else {
                    return {8, 64, 1};
                  }
                } else {
                  if (f1 <= 13.792481f) {
                    return {8, 32, 1};
                  } else {
                    return {8, 16, 1};
                  }
                }
              } else {
                if (f1 <= 11.292481f) {
                  if (f3 <= 6.500000f) {
                    if (f2 <= 4.500000f) {
                      if (f3 <= 3.500000f) {
                        return {8, 32, 1};
                      } else {
                        return {8, 32, 1};
                      }
                    } else {
                      if (f3 <= 1.500000f) {
                        return {8, 32, 2};
                      } else {
                        return {8, 32, 1};
                      }
                    }
                  } else {
                    return {8, 32, 1};
                  }
                } else {
                  if (f3 <= 7.500000f) {
                    if (f0 <= 4.292481f) {
                      return {8, 32, 1};
                    } else {
                      return {8, 32, 1};
                    }
                  } else {
                    if (f2 <= 4.500000f) {
                      if (f1 <= 13.792481f) {
                        return {8, 64, 1};
                      } else {
                        return {8, 16, 1};
                      }
                    } else {
                      if (f1 <= 13.292481f) {
                        return {8, 32, 1};
                      } else {
                        if (f0 <= 4.292481f) {
                          return {32, 32, 2};
                        } else {
                          return {16, 32, 2};
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if (f3 <= 5.500000f) {
                if (f0 <= 4.292481f) {
                  if (f3 <= 1.500000f) {
                    if (f1 <= 10.792481f) {
                      if (f2 <= 5.792481f) {
                        return {8, 16, 2};
                      } else {
                        return {8, 32, 4};
                      }
                    } else {
                      if (f3 <= 0.500000f) {
                        if (f1 <= 12.792481f) {
                          return {8, 32, 2};
                        } else {
                          return {8, 32, 1};
                        }
                      } else {
                        return {16, 32, 1};
                      }
                    }
                  } else {
                    if (f1 <= 12.292481f) {
                      if (f0 <= 3.292481f) {
                        return {16, 32, 1};
                      } else {
                        return {8, 32, 1};
                      }
                    } else {
                      if (f3 <= 4.500000f) {
                        return {16, 32, 2};
                      } else {
                        return {16, 64, 1};
                      }
                    }
                  }
                } else {
                  if (f1 <= 12.292481f) {
                    if (f3 <= 2.500000f) {
                      return {8, 32, 2};
                    } else {
                      return {8, 32, 1};
                    }
                  } else {
                    return {32, 32, 1};
                  }
                }
              } else {
                if (f1 <= 11.792481f) {
                  if (f3 <= 7.500000f) {
                    return {8, 32, 1};
                  } else {
                    if (f0 <= 3.292481f) {
                      return {8, 64, 1};
                    } else {
                      if (f1 <= 9.292481f) {
                        return {8, 32, 1};
                      } else {
                        return {16, 64, 1};
                      }
                    }
                  }
                } else {
                  if (f0 <= 4.292481f) {
                    if (f2 <= 5.792481f) {
                      if (f0 <= 3.292481f) {
                        return {8, 64, 2};
                      } else {
                        return {16, 64, 2};
                      }
                    } else {
                      if (f0 <= 3.292481f) {
                        return {32, 16, 4};
                      } else {
                        return {16, 64, 4};
                      }
                    }
                  } else {
                    if (f3 <= 7.500000f) {
                      if (f1 <= 14.792481f) {
                        return {16, 64, 1};
                      } else {
                        return {8, 64, 2};
                      }
                    } else {
                      if (f2 <= 5.792481f) {
                        return {16, 16, 2};
                      } else {
                        return {16, 16, 4};
                      }
                    }
                  }
                }
              }
            }
          } else {
            if (f2 <= 5.292481f) {
              if (f3 <= 2.500000f) {
                if (f1 <= 14.792481f) {
                  if (f2 <= 4.500000f) {
                    return {8, 32, 1};
                  } else {
                    return {8, 32, 1};
                  }
                } else {
                  return {16, 64, 1};
                }
              } else {
                if (f3 <= 6.500000f) {
                  if (f1 <= 9.792481f) {
                    if (f3 <= 5.500000f) {
                      return {8, 32, 1};
                    } else {
                      return {16, 64, 1};
                    }
                  } else {
                    return {16, 64, 1};
                  }
                } else {
                  if (f1 <= 12.792481f) {
                    return {16, 64, 1};
                  } else {
                    if (f2 <= 4.500000f) {
                      return {16, 32, 1};
                    } else {
                      return {16, 32, 2};
                    }
                  }
                }
              }
            } else {
              if (f3 <= 7.500000f) {
                if (f1 <= 11.292481f) {
                  if (f3 <= 4.500000f) {
                    if (f3 <= 0.500000f) {
                      return {8, 32, 2};
                    } else {
                      return {16, 32, 1};
                    }
                  } else {
                    return {16, 64, 1};
                  }
                } else {
                  return {16, 64, 1};
                }
              } else {
                if (f1 <= 12.292481f) {
                  return {16, 64, 1};
                } else {
                  return {16, 16, 1};
                }
              }
            }
          }
        } else {
          if (f0 <= 4.792481f) {
            if (f0 <= 4.292481f) {
              if (f0 <= 3.292481f) {
                if (f3 <= 5.500000f) {
                  if (f1 <= 12.292481f) {
                    if (f3 <= 3.500000f) {
                      if (f2 <= 7.292481f) {
                        return {8, 32, 4};
                      } else {
                        return {16, 16, 8};
                      }
                    } else {
                      if (f2 <= 10.292481f) {
                        if (f1 <= 10.792481f) {
                          return {16, 16, 1};
                        } else {
                          return {16, 32, 1};
                        }
                      } else {
                        return {8, 64, 1};
                      }
                    }
                  } else {
                    if (f2 <= 11.292481f) {
                      if (f3 <= 1.500000f) {
                        if (f1 <= 14.792481f) {
                          return {16, 16, 1};
                        } else {
                          return {8, 64, 1};
                        }
                      } else {
                        if (f2 <= 10.292481f) {
                          return {8, 64, 2};
                        } else {
                          return {8, 64, 2};
                        }
                      }
                    } else {
                      if (f3 <= 1.500000f) {
                        return {16, 64, 1};
                      } else {
                        return {8, 64, 1};
                      }
                    }
                  }
                } else {
                  if (f1 <= 11.792481f) {
                    if (f3 <= 8.500000f) {
                      if (f2 <= 10.292481f) {
                        return {8, 64, 1};
                      } else {
                        return {8, 64, 1};
                      }
                    } else {
                      if (f2 <= 8.292481f) {
                        return {8, 64, 2};
                      } else {
                        if (f1 <= 9.292481f) {
                          return {8, 64, 1};
                        } else {
                          return {8, 64, 8};
                        }
                      }
                    }
                  } else {
                    if (f2 <= 10.292481f) {
                      if (f3 <= 8.500000f) {
                        if (f2 <= 9.292481f) {
                          return {8, 64, 2};
                        } else {
                          return {8, 64, 4};
                        }
                      } else {
                        if (f1 <= 13.792481f) {
                          return {8, 64, 4};
                        } else {
                          return {8, 16, 4};
                        }
                      }
                    } else {
                      if (f1 <= 14.292481f) {
                        if (f2 <= 11.292481f) {
                          return {8, 64, 4};
                        } else {
                          return {64, 16, 4};
                        }
                      } else {
                        if (f3 <= 7.500000f) {
                          return {16, 64, 2};
                        } else {
                          return {64, 16, 2};
                        }
                      }
                    }
                  }
                }
              } else {
                if (f2 <= 10.292481f) {
                  if (f3 <= 6.500000f) {
                    if (f1 <= 11.292481f) {
                      if (f3 <= 3.500000f) {
                        if (f2 <= 7.292481f) {
                          return {8, 32, 4};
                        } else {
                          return {16, 16, 8};
                        }
                      } else {
                        if (f2 <= 7.292481f) {
                          return {16, 64, 1};
                        } else {
                          return {16, 16, 1};
                        }
                      }
                    } else {
                      if (f3 <= 1.500000f) {
                        if (f1 <= 13.792481f) {
                          return {16, 16, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        if (f2 <= 8.292481f) {
                          return {16, 64, 2};
                        } else {
                          return {16, 64, 4};
                        }
                      }
                    }
                  } else {
                    if (f1 <= 11.792481f) {
                      if (f2 <= 7.292481f) {
                        if (f3 <= 8.500000f) {
                          return {16, 64, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        if (f2 <= 8.292481f) {
                          return {8, 64, 2};
                        } else {
                          return {8, 64, 4};
                        }
                      }
                    } else {
                      if (f1 <= 14.792481f) {
                        if (f2 <= 8.292481f) {
                          return {16, 64, 4};
                        } else {
                          return {16, 64, 4};
                        }
                      } else {
                        if (f3 <= 7.500000f) {
                          return {16, 64, 4};
                        } else {
                          return {16, 16, 8};
                        }
                      }
                    }
                  }
                } else {
                  if (f1 <= 10.292481f) {
                    if (f3 <= 3.500000f) {
                      if (f2 <= 11.292481f) {
                        return {16, 16, 8};
                      } else {
                        if (f3 <= 1.500000f) {
                          return {16, 16, 8};
                        } else {
                          return {16, 64, 8};
                        }
                      }
                    } else {
                      if (f3 <= 4.500000f) {
                        return {16, 64, 8};
                      } else {
                        if (f3 <= 6.500000f) {
                          return {16, 64, 1};
                        } else {
                          return {8, 64, 4};
                        }
                      }
                    }
                  } else {
                    if (f3 <= 8.500000f) {
                      if (f3 <= 1.500000f) {
                        if (f1 <= 13.792481f) {
                          return {16, 16, 8};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        if (f1 <= 14.292481f) {
                          return {16, 64, 2};
                        } else {
                          return {16, 64, 2};
                        }
                      }
                    } else {
                      if (f1 <= 14.292481f) {
                        if (f1 <= 12.792481f) {
                          return {16, 64, 4};
                        } else {
                          return {32, 16, 8};
                        }
                      } else {
                        return {64, 16, 2};
                      }
                    }
                  }
                }
              }
            } else {
              if (f1 <= 13.292481f) {
                if (f3 <= 3.500000f) {
                  if (f1 <= 12.792481f) {
                    if (f2 <= 7.292481f) {
                      return {8, 32, 4};
                    } else {
                      if (f1 <= 10.292481f) {
                        return {8, 32, 8};
                      } else {
                        return {32, 32, 4};
                      }
                    }
                  } else {
                    return {8, 64, 4};
                  }
                } else {
                  if (f2 <= 8.292481f) {
                    return {8, 64, 4};
                  } else {
                    if (f2 <= 11.292481f) {
                      if (f3 <= 6.500000f) {
                        if (f1 <= 10.792481f) {
                          return {32, 32, 4};
                        } else {
                          return {8, 64, 4};
                        }
                      } else {
                        if (f1 <= 11.292481f) {
                          return {8, 64, 4};
                        } else {
                          return {8, 64, 4};
                        }
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              } else {
                if (f2 <= 10.292481f) {
                  if (f3 <= 7.500000f) {
                    if (f1 <= 13.792481f) {
                      return {32, 32, 4};
                    } else {
                      if (f1 <= 14.292481f) {
                        return {8, 64, 4};
                      } else {
                        return {8, 64, 2};
                      }
                    }
                  } else {
                    if (f2 <= 9.292481f) {
                      return {16, 16, 4};
                    } else {
                      return {32, 16, 8};
                    }
                  }
                } else {
                  if (f3 <= 8.500000f) {
                    if (f2 <= 11.292481f) {
                      return {32, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    if (f1 <= 14.292481f) {
                      return {64, 16, 4};
                    } else {
                      return {64, 16, 2};
                    }
                  }
                }
              }
            }
          } else {
            if (f2 <= 8.292481f) {
              if (f0 <= 5.292481f) {
                if (f3 <= 5.500000f) {
                  if (f1 <= 11.792481f) {
                    return {16, 32, 4};
                  } else {
                    if (f3 <= 1.500000f) {
                      return {32, 32, 1};
                    } else {
                      if (f2 <= 7.292481f) {
                        return {32, 64, 1};
                      } else {
                        return {32, 32, 2};
                      }
                    }
                  }
                } else {
                  if (f1 <= 13.792481f) {
                    if (f2 <= 7.292481f) {
                      return {32, 64, 1};
                    } else {
                      return {16, 64, 4};
                    }
                  } else {
                    if (f3 <= 7.500000f) {
                      return {16, 64, 2};
                    } else {
                      return {16, 16, 4};
                    }
                  }
                }
              } else {
                if (f3 <= 4.500000f) {
                  if (f1 <= 11.792481f) {
                    if (f3 <= 3.500000f) {
                      return {16, 32, 4};
                    } else {
                      if (f1 <= 10.292481f) {
                        return {16, 32, 4};
                      } else {
                        return {16, 64, 1};
                      }
                    }
                  } else {
                    return {16, 64, 1};
                  }
                } else {
                  if (f1 <= 12.292481f) {
                    if (f1 <= 10.792481f) {
                      return {16, 64, 1};
                    } else {
                      if (f3 <= 8.500000f) {
                        return {16, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  } else {
                    if (f3 <= 7.500000f) {
                      if (f1 <= 14.292481f) {
                        return {16, 64, 1};
                      } else {
                        return {16, 64, 2};
                      }
                    } else {
                      if (f2 <= 7.292481f) {
                        if (f1 <= 14.292481f) {
                          return {16, 16, 1};
                        } else {
                          return {16, 16, 4};
                        }
                      } else {
                        return {16, 16, 1};
                      }
                    }
                  }
                }
              }
            } else {
              if (f3 <= 5.500000f) {
                if (f1 <= 12.792481f) {
                  if (f3 <= 2.500000f) {
                    if (f1 <= 10.292481f) {
                      if (f2 <= 9.292481f) {
                        return {16, 32, 8};
                      } else {
                        if (f3 <= 1.500000f) {
                          return {16, 16, 8};
                        } else {
                          return {32, 16, 8};
                        }
                      }
                    } else {
                      if (f2 <= 11.292481f) {
                        if (f3 <= 0.500000f) {
                          return {16, 32, 8};
                        } else {
                          return {32, 32, 4};
                        }
                      } else {
                        return {32, 64, 8};
                      }
                    }
                  } else {
                    if (f2 <= 11.292481f) {
                      if (f1 <= 10.792481f) {
                        return {32, 32, 4};
                      } else {
                        if (f0 <= 5.292481f) {
                          return {32, 32, 4};
                        } else {
                          return {32, 32, 2};
                        }
                      }
                    } else {
                      if (f1 <= 11.292481f) {
                        if (f3 <= 4.500000f) {
                          return {32, 64, 8};
                        } else {
                          return {16, 64, 2};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  }
                } else {
                  if (f2 <= 10.292481f) {
                    if (f0 <= 5.292481f) {
                      if (f3 <= 2.500000f) {
                        if (f1 <= 14.792481f) {
                          return {32, 32, 4};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        return {16, 64, 2};
                      }
                    } else {
                      if (f0 <= 5.792481f) {
                        if (f3 <= 2.500000f) {
                          return {16, 64, 1};
                        } else {
                          return {16, 64, 2};
                        }
                      } else {
                        if (f2 <= 9.292481f) {
                          return {64, 32, 2};
                        } else {
                          return {64, 32, 4};
                        }
                      }
                    }
                  } else {
                    if (f0 <= 5.792481f) {
                      if (f0 <= 5.292481f) {
                        if (f2 <= 11.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        if (f3 <= 1.500000f) {
                          return {16, 64, 1};
                        } else {
                          return {16, 64, 2};
                        }
                      }
                    } else {
                      if (f3 <= 3.500000f) {
                        return {32, 64, 1};
                      } else {
                        return {64, 32, 2};
                      }
                    }
                  }
                }
              } else {
                if (f1 <= 11.292481f) {
                  if (f0 <= 5.292481f) {
                    if (f2 <= 11.292481f) {
                      if (f3 <= 8.500000f) {
                        if (f1 <= 9.792481f) {
                          return {8, 64, 4};
                        } else {
                          return {16, 64, 2};
                        }
                      } else {
                        if (f2 <= 10.292481f) {
                          return {16, 64, 4};
                        } else {
                          return {32, 64, 8};
                        }
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    if (f3 <= 8.500000f) {
                      if (f2 <= 11.292481f) {
                        if (f2 <= 9.292481f) {
                          return {16, 64, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      if (f1 <= 9.792481f) {
                        if (f2 <= 9.292481f) {
                          return {16, 64, 4};
                        } else {
                          return {64, 32, 4};
                        }
                      } else {
                        if (f0 <= 5.792481f) {
                          return {16, 64, 4};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  }
                } else {
                  if (f2 <= 10.292481f) {
                    if (f2 <= 9.292481f) {
                      if (f0 <= 5.292481f) {
                        if (f1 <= 13.792481f) {
                          return {16, 64, 4};
                        } else {
                          return {16, 16, 4};
                        }
                      } else {
                        if (f1 <= 14.292481f) {
                          return {16, 64, 1};
                        } else {
                          return {16, 16, 1};
                        }
                      }
                    } else {
                      if (f1 <= 14.292481f) {
                        if (f3 <= 7.500000f) {
                          return {64, 32, 4};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        if (f3 <= 7.500000f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 16, 2};
                        }
                      }
                    }
                  } else {
                    if (f3 <= 8.500000f) {
                      if (f1 <= 15.792481f) {
                        return {32, 64, 1};
                      } else {
                        if (f3 <= 6.500000f) {
                          return {32, 64, 1};
                        } else {
                          return {64, 16, 2};
                        }
                      }
                    } else {
                      if (f1 <= 13.292481f) {
                        return {32, 64, 1};
                      } else {
                        if (f0 <= 5.292481f) {
                          return {64, 16, 2};
                        } else {
                          return {64, 16, 4};
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if (f3 <= 3.500000f) {
          if (f1 <= 12.792481f) {
            if (f1 <= 10.292481f) {
              if (f3 <= 1.500000f) {
                if (f0 <= 5.292481f) {
                  if (f1 <= 9.792481f) {
                    return {16, 16, 8};
                  } else {
                    if (f3 <= 0.500000f) {
                      return {16, 16, 8};
                    } else {
                      return {8, 64, 8};
                    }
                  }
                } else {
                  if (f3 <= 0.500000f) {
                    if (f1 <= 9.292481f) {
                      return {16, 16, 8};
                    } else {
                      return {16, 32, 8};
                    }
                  } else {
                    return {16, 64, 8};
                  }
                }
              } else {
                if (f0 <= 3.292481f) {
                  return {8, 64, 8};
                } else {
                  if (f0 <= 4.792481f) {
                    return {16, 64, 8};
                  } else {
                    return {16, 64, 8};
                  }
                }
              }
            } else {
              if (f0 <= 4.292481f) {
                if (f0 <= 3.292481f) {
                  if (f3 <= 1.500000f) {
                    return {8, 64, 8};
                  } else {
                    return {8, 64, 2};
                  }
                } else {
                  if (f3 <= 1.500000f) {
                    if (f2 <= 14.292481f) {
                      if (f2 <= 13.292481f) {
                        return {16, 64, 8};
                      } else {
                        return {8, 64, 8};
                      }
                    } else {
                      if (f1 <= 10.792481f) {
                        return {16, 16, 8};
                      } else {
                        return {16, 64, 8};
                      }
                    }
                  } else {
                    if (f1 <= 11.792481f) {
                      if (f2 <= 14.292481f) {
                        return {16, 64, 4};
                      } else {
                        if (f3 <= 2.500000f) {
                          return {16, 64, 8};
                        } else {
                          return {16, 64, 4};
                        }
                      }
                    } else {
                      if (f3 <= 2.500000f) {
                        if (f2 <= 14.292481f) {
                          return {16, 64, 2};
                        } else {
                          return {16, 64, 4};
                        }
                      } else {
                        return {16, 64, 1};
                      }
                    }
                  }
                }
              } else {
                if (f3 <= 1.500000f) {
                  if (f1 <= 11.292481f) {
                    return {16, 64, 8};
                  } else {
                    return {32, 64, 8};
                  }
                } else {
                  if (f1 <= 11.792481f) {
                    if (f0 <= 5.292481f) {
                      if (f2 <= 14.292481f) {
                        return {16, 64, 2};
                      } else {
                        return {32, 64, 8};
                      }
                    } else {
                      return {64, 32, 2};
                    }
                  } else {
                    if (f3 <= 2.500000f) {
                      return {32, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              }
            }
          } else {
            if (f0 <= 4.292481f) {
              if (f3 <= 1.500000f) {
                if (f1 <= 13.792481f) {
                  return {16, 64, 4};
                } else {
                  if (f2 <= 14.292481f) {
                    return {16, 64, 1};
                  } else {
                    if (f1 <= 15.792481f) {
                      return {16, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              } else {
                if (f2 <= 13.292481f) {
                  if (f1 <= 14.792481f) {
                    return {16, 64, 1};
                  } else {
                    return {16, 64, 2};
                  }
                } else {
                  if (f1 <= 13.792481f) {
                    if (f3 <= 2.500000f) {
                      return {16, 64, 1};
                    } else {
                      if (f2 <= 14.292481f) {
                        return {16, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  } else {
                    return {32, 64, 1};
                  }
                }
              }
            } else {
              if (f3 <= 0.500000f) {
                if (f1 <= 14.792481f) {
                  if (f1 <= 13.792481f) {
                    if (f0 <= 5.292481f) {
                      return {32, 64, 8};
                    } else {
                      return {64, 32, 2};
                    }
                  } else {
                    if (f0 <= 5.792481f) {
                      return {16, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  }
                } else {
                  return {32, 64, 1};
                }
              } else {
                if (f1 <= 13.292481f) {
                  return {32, 64, 1};
                } else {
                  return {32, 64, 1};
                }
              }
            }
          }
        } else {
          if (f1 <= 10.292481f) {
            if (f0 <= 3.292481f) {
              if (f3 <= 5.500000f) {
                return {8, 64, 2};
              } else {
                if (f3 <= 9.500000f) {
                  return {8, 64, 1};
                } else {
                  if (f1 <= 8.792481f) {
                    return {8, 64, 1};
                  } else {
                    return {8, 16, 4};
                  }
                }
              }
            } else {
              if (f3 <= 5.500000f) {
                if (f0 <= 4.292481f) {
                  if (f1 <= 9.792481f) {
                    return {16, 64, 8};
                  } else {
                    if (f3 <= 4.500000f) {
                      return {16, 64, 2};
                    } else {
                      return {16, 64, 1};
                    }
                  }
                } else {
                  if (f0 <= 5.292481f) {
                    return {16, 64, 4};
                  } else {
                    return {32, 64, 1};
                  }
                }
              } else {
                if (f0 <= 4.292481f) {
                  if (f2 <= 15.292481f) {
                    if (f3 <= 8.500000f) {
                      return {16, 64, 2};
                    } else {
                      if (f1 <= 9.292481f) {
                        return {16, 64, 8};
                      } else {
                        return {8, 16, 8};
                      }
                    }
                  } else {
                    return {32, 64, 1};
                  }
                } else {
                  if (f2 <= 15.292481f) {
                    if (f3 <= 8.500000f) {
                      if (f2 <= 13.292481f) {
                        if (f3 <= 7.500000f) {
                          return {16, 64, 2};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    return {32, 64, 1};
                  }
                }
              }
            }
          } else {
            if (f3 <= 8.500000f) {
              if (f2 <= 13.292481f) {
                if (f0 <= 4.292481f) {
                  if (f1 <= 12.292481f) {
                    if (f3 <= 7.500000f) {
                      return {16, 64, 2};
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    if (f3 <= 5.500000f) {
                      if (f1 <= 13.792481f) {
                        return {16, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                } else {
                  return {32, 64, 1};
                }
              } else {
                if (f1 <= 12.292481f) {
                  if (f3 <= 4.500000f) {
                    if (f0 <= 4.292481f) {
                      if (f2 <= 14.292481f) {
                        return {16, 64, 2};
                      } else {
                        if (f1 <= 11.792481f) {
                          return {16, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    return {32, 64, 1};
                  }
                } else {
                  return {32, 64, 1};
                }
              }
            } else {
              if (f1 <= 14.792481f) {
                if (f2 <= 14.292481f) {
                  if (f1 <= 12.292481f) {
                    if (f0 <= 5.292481f) {
                      if (f0 <= 4.292481f) {
                        if (f0 <= 3.292481f) {
                          return {8, 16, 4};
                        } else {
                          return {8, 32, 8};
                        }
                      } else {
                        return {32, 32, 4};
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    if (f3 <= 9.500000f) {
                      return {32, 64, 1};
                    } else {
                      if (f1 <= 13.792481f) {
                        return {32, 64, 1};
                      } else {
                        if (f0 <= 5.292481f) {
                          return {64, 16, 2};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  }
                } else {
                  return {32, 64, 1};
                }
              } else {
                if (f2 <= 15.292481f) {
                  if (f0 <= 5.292481f) {
                    return {64, 16, 2};
                  } else {
                    if (f2 <= 13.292481f) {
                      if (f3 <= 9.500000f) {
                        return {64, 16, 4};
                      } else {
                        return {64, 16, 1};
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                } else {
                  return {32, 64, 1};
                }
              }
            }
          }
        }
      }
    } else {
      if (f2 <= 10.292481f) {
        if (f3 <= 8.500000f) {
          if (f3 <= 2.500000f) {
            if (f2 <= 9.292481f) {
              if (f1 <= 11.292481f) {
                if (f0 <= 9.292481f) {
                  if (f2 <= 6.292481f) {
                    if (f2 <= 4.500000f) {
                      if (f1 <= 10.292481f) {
                        return {8, 32, 1};
                      } else {
                        return {16, 64, 1};
                      }
                    } else {
                      if (f1 <= 10.292481f) {
                        if (f2 <= 5.292481f) {
                          return {16, 32, 1};
                        } else {
                          return {16, 32, 1};
                        }
                      } else {
                        if (f3 <= 1.500000f) {
                          return {16, 32, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      }
                    }
                  } else {
                    if (f2 <= 8.292481f) {
                      if (f1 <= 10.292481f) {
                        if (f3 <= 1.500000f) {
                          return {16, 32, 4};
                        } else {
                          return {16, 32, 4};
                        }
                      } else {
                        if (f3 <= 0.500000f) {
                          return {16, 32, 4};
                        } else {
                          return {16, 64, 1};
                        }
                      }
                    } else {
                      if (f3 <= 1.500000f) {
                        if (f1 <= 10.292481f) {
                          return {16, 32, 8};
                        } else {
                          return {32, 32, 4};
                        }
                      } else {
                        return {32, 32, 4};
                      }
                    }
                  }
                } else {
                  if (f2 <= 8.292481f) {
                    if (f1 <= 9.792481f) {
                      if (f0 <= 10.292481f) {
                        if (f3 <= 0.500000f) {
                          return {16, 32, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        if (f3 <= 1.500000f) {
                          return {16, 64, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      }
                    } else {
                      if (f0 <= 10.792481f) {
                        return {16, 64, 1};
                      } else {
                        if (f3 <= 0.500000f) {
                          return {16, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  } else {
                    if (f0 <= 11.292481f) {
                      if (f3 <= 1.500000f) {
                        return {32, 32, 4};
                      } else {
                        return {32, 32, 2};
                      }
                    } else {
                      return {16, 64, 1};
                    }
                  }
                }
              } else {
                if (f0 <= 8.792481f) {
                  if (f2 <= 8.292481f) {
                    if (f0 <= 7.292481f) {
                      if (f1 <= 12.292481f) {
                        if (f2 <= 6.292481f) {
                          return {32, 32, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        if (f2 <= 7.292481f) {
                          return {16, 64, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      }
                    } else {
                      if (f2 <= 6.292481f) {
                        if (f2 <= 5.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        if (f1 <= 13.292481f) {
                          return {16, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  } else {
                    if (f1 <= 13.292481f) {
                      if (f3 <= 1.500000f) {
                        return {32, 32, 4};
                      } else {
                        return {64, 32, 2};
                      }
                    } else {
                      return {16, 64, 1};
                    }
                  }
                } else {
                  if (f2 <= 6.292481f) {
                    if (f1 <= 14.792481f) {
                      if (f2 <= 5.292481f) {
                        return {32, 64, 1};
                      } else {
                        return {16, 64, 1};
                      }
                    } else {
                      if (f0 <= 9.792481f) {
                        return {16, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  } else {
                    if (f2 <= 8.292481f) {
                      return {32, 64, 1};
                    } else {
                      if (f1 <= 14.292481f) {
                        if (f1 <= 12.292481f) {
                          return {32, 32, 2};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        if (f0 <= 10.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if (f1 <= 12.292481f) {
                if (f0 <= 9.292481f) {
                  if (f1 <= 9.792481f) {
                    if (f3 <= 1.500000f) {
                      if (f0 <= 8.292481f) {
                        return {16, 32, 8};
                      } else {
                        return {32, 32, 4};
                      }
                    } else {
                      return {32, 32, 4};
                    }
                  } else {
                    if (f3 <= 1.500000f) {
                      return {32, 32, 4};
                    } else {
                      if (f0 <= 7.792481f) {
                        return {32, 32, 4};
                      } else {
                        return {64, 32, 4};
                      }
                    }
                  }
                } else {
                  if (f3 <= 0.500000f) {
                    if (f1 <= 10.292481f) {
                      return {32, 32, 4};
                    } else {
                      return {64, 32, 4};
                    }
                  } else {
                    return {64, 32, 4};
                  }
                }
              } else {
                if (f0 <= 10.792481f) {
                  if (f0 <= 6.792481f) {
                    return {32, 64, 1};
                  } else {
                    return {64, 32, 4};
                  }
                } else {
                  if (f1 <= 13.792481f) {
                    if (f3 <= 1.500000f) {
                      return {64, 32, 4};
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    return {32, 64, 1};
                  }
                }
              }
            }
          } else {
            if (f0 <= 7.292481f) {
              if (f1 <= 12.292481f) {
                if (f2 <= 8.292481f) {
                  if (f2 <= 7.292481f) {
                    if (f3 <= 4.500000f) {
                      if (f2 <= 6.292481f) {
                        return {32, 32, 1};
                      } else {
                        return {16, 64, 1};
                      }
                    } else {
                      if (f1 <= 11.292481f) {
                        return {16, 64, 1};
                      } else {
                        if (f3 <= 7.500000f) {
                          return {16, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  } else {
                    if (f3 <= 4.500000f) {
                      return {32, 32, 2};
                    } else {
                      return {16, 64, 1};
                    }
                  }
                } else {
                  if (f3 <= 5.500000f) {
                    if (f1 <= 10.292481f) {
                      if (f3 <= 4.500000f) {
                        return {32, 32, 4};
                      } else {
                        return {32, 32, 2};
                      }
                    } else {
                      return {32, 32, 2};
                    }
                  } else {
                    if (f2 <= 9.292481f) {
                      return {16, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              } else {
                if (f3 <= 4.500000f) {
                  if (f2 <= 9.292481f) {
                    if (f1 <= 14.792481f) {
                      if (f2 <= 7.292481f) {
                        return {16, 64, 1};
                      } else {
                        if (f2 <= 8.292481f) {
                          return {32, 32, 1};
                        } else {
                          return {16, 64, 1};
                        }
                      }
                    } else {
                      if (f3 <= 3.500000f) {
                        return {16, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  } else {
                    if (f1 <= 14.792481f) {
                      return {64, 32, 4};
                    } else {
                      return {32, 64, 1};
                    }
                  }
                } else {
                  if (f1 <= 14.292481f) {
                    if (f3 <= 5.500000f) {
                      return {32, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    if (f3 <= 6.500000f) {
                      if (f2 <= 5.292481f) {
                        return {32, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      if (f2 <= 9.292481f) {
                        if (f2 <= 5.292481f) {
                          return {16, 32, 1};
                        } else {
                          return {16, 16, 1};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  }
                }
              }
            } else {
              if (f2 <= 6.292481f) {
                if (f2 <= 4.500000f) {
                  if (f1 <= 14.292481f) {
                    if (f3 <= 6.500000f) {
                      if (f1 <= 9.292481f) {
                        return {16, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      if (f0 <= 11.292481f) {
                        if (f1 <= 11.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {16, 32, 1};
                        }
                      } else {
                        return {16, 64, 1};
                      }
                    }
                  } else {
                    if (f3 <= 4.500000f) {
                      return {16, 32, 1};
                    } else {
                      if (f0 <= 11.292481f) {
                        return {16, 32, 1};
                      } else {
                        return {16, 32, 1};
                      }
                    }
                  }
                } else {
                  if (f3 <= 6.500000f) {
                    if (f1 <= 10.292481f) {
                      if (f2 <= 5.292481f) {
                        if (f0 <= 10.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        return {16, 64, 1};
                      }
                    } else {
                      if (f2 <= 5.292481f) {
                        if (f1 <= 14.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {16, 32, 2};
                        }
                      } else {
                        if (f3 <= 5.500000f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  } else {
                    if (f1 <= 11.292481f) {
                      if (f0 <= 8.792481f) {
                        if (f1 <= 9.792481f) {
                          return {16, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      if (f2 <= 5.292481f) {
                        if (f0 <= 8.792481f) {
                          return {16, 32, 2};
                        } else {
                          return {16, 32, 2};
                        }
                      } else {
                        if (f0 <= 9.292481f) {
                          return {16, 16, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  }
                }
              } else {
                if (f1 <= 10.292481f) {
                  if (f3 <= 5.500000f) {
                    if (f2 <= 8.292481f) {
                      if (f0 <= 9.792481f) {
                        if (f2 <= 7.292481f) {
                          return {16, 64, 1};
                        } else {
                          return {32, 32, 2};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      if (f2 <= 9.292481f) {
                        if (f0 <= 10.292481f) {
                          return {32, 32, 2};
                        } else {
                          return {16, 64, 1};
                        }
                      } else {
                        if (f0 <= 8.292481f) {
                          return {32, 32, 4};
                        } else {
                          return {64, 32, 4};
                        }
                      }
                    }
                  } else {
                    if (f0 <= 8.792481f) {
                      if (f2 <= 8.292481f) {
                        if (f3 <= 6.500000f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        if (f2 <= 9.292481f) {
                          return {16, 64, 1};
                        } else {
                          return {64, 32, 4};
                        }
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                } else {
                  if (f3 <= 7.500000f) {
                    if (f3 <= 5.500000f) {
                      if (f2 <= 8.292481f) {
                        if (f1 <= 14.792481f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        if (f1 <= 12.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    } else {
                      if (f2 <= 8.292481f) {
                        if (f2 <= 7.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  } else {
                    if (f2 <= 9.292481f) {
                      if (f2 <= 8.292481f) {
                        if (f0 <= 10.292481f) {
                          return {16, 16, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        if (f0 <= 10.292481f) {
                          return {16, 16, 4};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              }
            }
          }
        } else {
          if (f2 <= 8.292481f) {
            if (f2 <= 6.292481f) {
              if (f2 <= 4.500000f) {
                if (f0 <= 11.292481f) {
                  if (f1 <= 9.292481f) {
                    return {16, 64, 1};
                  } else {
                    if (f1 <= 11.292481f) {
                      if (f0 <= 8.292481f) {
                        return {8, 64, 1};
                      } else {
                        return {16, 32, 1};
                      }
                    } else {
                      return {16, 32, 1};
                    }
                  }
                } else {
                  if (f1 <= 9.292481f) {
                    return {8, 16, 1};
                  } else {
                    return {16, 64, 1};
                  }
                }
              } else {
                if (f0 <= 9.292481f) {
                  if (f2 <= 5.292481f) {
                    if (f1 <= 10.792481f) {
                      return {32, 64, 1};
                    } else {
                      return {16, 32, 2};
                    }
                  } else {
                    if (f1 <= 11.792481f) {
                      return {32, 64, 1};
                    } else {
                      return {16, 16, 1};
                    }
                  }
                } else {
                  if (f1 <= 9.292481f) {
                    if (f2 <= 5.292481f) {
                      return {8, 16, 1};
                    } else {
                      if (f0 <= 10.792481f) {
                        return {32, 64, 1};
                      } else {
                        return {16, 16, 1};
                      }
                    }
                  } else {
                    return {16, 32, 2};
                  }
                }
              }
            } else {
              if (f0 <= 11.292481f) {
                if (f1 <= 10.792481f) {
                  if (f0 <= 9.792481f) {
                    if (f0 <= 7.292481f) {
                      if (f1 <= 9.792481f) {
                        return {16, 64, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    if (f1 <= 9.292481f) {
                      if (f3 <= 9.500000f) {
                        return {32, 64, 1};
                      } else {
                        return {16, 16, 1};
                      }
                    } else {
                      return {16, 16, 1};
                    }
                  }
                } else {
                  if (f0 <= 10.292481f) {
                    if (f1 <= 11.792481f) {
                      if (f0 <= 8.292481f) {
                        if (f3 <= 9.500000f) {
                          return {32, 64, 1};
                        } else {
                          return {16, 16, 1};
                        }
                      } else {
                        return {16, 16, 1};
                      }
                    } else {
                      if (f0 <= 9.292481f) {
                        return {16, 16, 1};
                      } else {
                        if (f1 <= 14.292481f) {
                          return {16, 16, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    }
                  } else {
                    if (f2 <= 7.292481f) {
                      if (f1 <= 12.292481f) {
                        return {16, 16, 1};
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      if (f1 <= 14.292481f) {
                        if (f3 <= 9.500000f) {
                          return {16, 16, 1};
                        } else {
                          return {16, 16, 2};
                        }
                      } else {
                        return {16, 16, 2};
                      }
                    }
                  }
                }
              } else {
                if (f2 <= 7.292481f) {
                  if (f1 <= 10.292481f) {
                    if (f1 <= 9.292481f) {
                      return {16, 16, 2};
                    } else {
                      return {16, 16, 1};
                    }
                  } else {
                    return {32, 64, 1};
                  }
                } else {
                  if (f1 <= 11.292481f) {
                    if (f3 <= 9.500000f) {
                      return {16, 16, 1};
                    } else {
                      return {16, 16, 2};
                    }
                  } else {
                    return {16, 16, 2};
                  }
                }
              }
            }
          } else {
            if (f0 <= 9.292481f) {
              if (f1 <= 11.792481f) {
                return {32, 64, 1};
              } else {
                if (f2 <= 9.292481f) {
                  if (f0 <= 8.292481f) {
                    return {16, 16, 1};
                  } else {
                    return {16, 16, 4};
                  }
                } else {
                  if (f0 <= 7.292481f) {
                    if (f1 <= 14.292481f) {
                      return {32, 16, 4};
                    } else {
                      return {64, 16, 1};
                    }
                  } else {
                    return {32, 16, 4};
                  }
                }
              }
            } else {
              if (f2 <= 9.292481f) {
                if (f1 <= 10.292481f) {
                  if (f0 <= 11.292481f) {
                    if (f1 <= 9.292481f) {
                      return {32, 64, 1};
                    } else {
                      return {16, 16, 4};
                    }
                  } else {
                    return {16, 16, 2};
                  }
                } else {
                  return {16, 16, 4};
                }
              } else {
                if (f3 <= 9.500000f) {
                  if (f1 <= 9.292481f) {
                    return {32, 64, 1};
                  } else {
                    if (f1 <= 14.292481f) {
                      return {16, 16, 4};
                    } else {
                      if (f0 <= 10.292481f) {
                        return {16, 16, 4};
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  }
                } else {
                  if (f1 <= 10.292481f) {
                    if (f1 <= 9.292481f) {
                      return {8, 16, 4};
                    } else {
                      return {16, 16, 4};
                    }
                  } else {
                    if (f0 <= 10.292481f) {
                      if (f1 <= 15.292481f) {
                        return {32, 16, 8};
                      } else {
                        return {32, 16, 4};
                      }
                    } else {
                      if (f1 <= 12.292481f) {
                        if (f0 <= 11.292481f) {
                          return {32, 16, 8};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if (f1 <= 10.292481f) {
          if (f3 <= 2.500000f) {
            if (f0 <= 10.292481f) {
              if (f2 <= 11.292481f) {
                if (f0 <= 7.792481f) {
                  if (f3 <= 1.500000f) {
                    return {16, 32, 8};
                  } else {
                    return {32, 32, 4};
                  }
                } else {
                  return {32, 32, 4};
                }
              } else {
                if (f2 <= 14.292481f) {
                  if (f2 <= 13.292481f) {
                    if (f3 <= 1.500000f) {
                      if (f0 <= 8.292481f) {
                        return {32, 32, 8};
                      } else {
                        if (f2 <= 12.292481f) {
                          return {16, 32, 8};
                        } else {
                          return {32, 32, 2};
                        }
                      }
                    } else {
                      if (f0 <= 8.292481f) {
                        return {32, 64, 8};
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  } else {
                    if (f0 <= 8.292481f) {
                      return {16, 64, 8};
                    } else {
                      return {64, 32, 2};
                    }
                  }
                } else {
                  if (f0 <= 8.292481f) {
                    if (f3 <= 1.500000f) {
                      return {16, 64, 8};
                    } else {
                      return {64, 32, 4};
                    }
                  } else {
                    if (f3 <= 1.500000f) {
                      return {64, 32, 4};
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              }
            } else {
              return {32, 64, 1};
            }
          } else {
            if (f3 <= 5.500000f) {
              if (f0 <= 7.792481f) {
                if (f2 <= 12.292481f) {
                  if (f3 <= 4.500000f) {
                    if (f2 <= 11.292481f) {
                      return {32, 32, 4};
                    } else {
                      return {32, 32, 2};
                    }
                  } else {
                    if (f2 <= 11.292481f) {
                      return {32, 64, 1};
                    } else {
                      return {16, 64, 2};
                    }
                  }
                } else {
                  if (f3 <= 3.500000f) {
                    return {32, 64, 1};
                  } else {
                    return {32, 64, 1};
                  }
                }
              } else {
                if (f2 <= 12.292481f) {
                  if (f0 <= 10.292481f) {
                    if (f3 <= 4.500000f) {
                      if (f0 <= 9.292481f) {
                        return {32, 32, 2};
                      } else {
                        if (f2 <= 11.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    } else {
                      return {64, 32, 4};
                    }
                  } else {
                    return {32, 64, 1};
                  }
                } else {
                  if (f0 <= 10.292481f) {
                    return {32, 64, 1};
                  } else {
                    if (f2 <= 13.292481f) {
                      return {32, 64, 1};
                    } else {
                      if (f2 <= 15.292481f) {
                        if (f0 <= 11.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {8, 64, 1};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  }
                }
              }
            } else {
              if (f0 <= 10.292481f) {
                if (f2 <= 11.292481f) {
                  if (f0 <= 7.292481f) {
                    if (f3 <= 7.500000f) {
                      return {16, 64, 1};
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    return {32, 64, 1};
                  }
                } else {
                  return {32, 64, 1};
                }
              } else {
                if (f3 <= 8.500000f) {
                  if (f2 <= 13.292481f) {
                    return {32, 64, 1};
                  } else {
                    if (f1 <= 9.292481f) {
                      if (f2 <= 15.292481f) {
                        if (f3 <= 7.500000f) {
                          return {8, 64, 1};
                        } else {
                          return {32, 64, 1};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                } else {
                  if (f2 <= 14.292481f) {
                    if (f2 <= 11.292481f) {
                      if (f1 <= 9.292481f) {
                        return {8, 16, 4};
                      } else {
                        if (f0 <= 11.292481f) {
                          return {16, 16, 4};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    } else {
                      if (f1 <= 9.292481f) {
                        if (f0 <= 11.292481f) {
                          return {32, 64, 1};
                        } else {
                          return {8, 64, 4};
                        }
                      } else {
                        return {32, 64, 1};
                      }
                    }
                  } else {
                    if (f0 <= 11.292481f) {
                      return {32, 64, 1};
                    } else {
                      return {8, 64, 1};
                    }
                  }
                }
              }
            }
          }
        } else {
          if (f3 <= 8.500000f) {
            if (f2 <= 12.292481f) {
              if (f0 <= 9.292481f) {
                if (f3 <= 7.500000f) {
                  if (f1 <= 11.792481f) {
                    if (f3 <= 2.500000f) {
                      if (f0 <= 7.292481f) {
                        return {32, 32, 4};
                      } else {
                        if (f2 <= 11.292481f) {
                          return {32, 32, 4};
                        } else {
                          return {32, 64, 1};
                        }
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    if (f3 <= 6.500000f) {
                      return {32, 64, 1};
                    } else {
                      if (f1 <= 14.792481f) {
                        return {32, 64, 1};
                      } else {
                        return {64, 16, 4};
                      }
                    }
                  }
                } else {
                  if (f1 <= 13.292481f) {
                    if (f1 <= 12.292481f) {
                      return {32, 64, 1};
                    } else {
                      if (f0 <= 7.792481f) {
                        return {32, 64, 1};
                      } else {
                        return {64, 16, 4};
                      }
                    }
                  } else {
                    return {64, 16, 4};
                  }
                }
              } else {
                if (f0 <= 11.292481f) {
                  return {32, 64, 1};
                } else {
                  if (f1 <= 13.792481f) {
                    return {32, 64, 1};
                  } else {
                    if (f3 <= 5.500000f) {
                      if (f3 <= 2.500000f) {
                        return {32, 64, 1};
                      } else {
                        if (f2 <= 11.292481f) {
                          return {16, 64, 2};
                        } else {
                          return {16, 64, 4};
                        }
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              }
            } else {
              if (f3 <= 0.500000f) {
                if (f0 <= 8.292481f) {
                  if (f1 <= 12.292481f) {
                    return {32, 32, 2};
                  } else {
                    return {32, 64, 1};
                  }
                } else {
                  return {32, 64, 1};
                }
              } else {
                return {32, 64, 1};
              }
            }
          } else {
            if (f2 <= 13.292481f) {
              if (f0 <= 9.292481f) {
                if (f1 <= 11.792481f) {
                  if (f0 <= 8.292481f) {
                    return {32, 64, 1};
                  } else {
                    if (f3 <= 9.500000f) {
                      if (f1 <= 10.792481f) {
                        return {32, 64, 1};
                      } else {
                        return {64, 16, 4};
                      }
                    } else {
                      return {64, 16, 4};
                    }
                  }
                } else {
                  if (f0 <= 7.292481f) {
                    if (f1 <= 12.792481f) {
                      if (f3 <= 9.500000f) {
                        return {32, 64, 1};
                      } else {
                        return {64, 16, 4};
                      }
                    } else {
                      return {64, 16, 4};
                    }
                  } else {
                    return {64, 16, 4};
                  }
                }
              } else {
                if (f1 <= 12.292481f) {
                  if (f0 <= 11.292481f) {
                    if (f2 <= 12.292481f) {
                      return {64, 16, 4};
                    } else {
                      return {32, 64, 1};
                    }
                  } else {
                    return {32, 64, 1};
                  }
                } else {
                  if (f2 <= 12.292481f) {
                    return {32, 64, 1};
                  } else {
                    if (f0 <= 10.292481f) {
                      if (f1 <= 13.292481f) {
                        return {32, 64, 1};
                      } else {
                        return {64, 32, 4};
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              }
            } else {
              if (f2 <= 14.292481f) {
                if (f1 <= 14.292481f) {
                  if (f0 <= 10.292481f) {
                    return {32, 64, 1};
                  } else {
                    if (f3 <= 9.500000f) {
                      if (f1 <= 12.292481f) {
                        return {32, 64, 1};
                      } else {
                        return {16, 64, 4};
                      }
                    } else {
                      return {32, 64, 1};
                    }
                  }
                } else {
                  if (f0 <= 8.292481f) {
                    return {32, 64, 1};
                  } else {
                    if (f0 <= 10.292481f) {
                      return {64, 32, 4};
                    } else {
                      return {32, 64, 1};
                    }
                  }
                }
              } else {
                return {32, 64, 1};
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace onnxruntime::webgpu::intel::xe3lpg
